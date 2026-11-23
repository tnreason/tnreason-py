"""Step 1: boosting loop over chain architectures with filtered link prediction.

Sparse form of contracting the selection architecture against the data: candidate bodies are
priced against every head relation in one pass over the fact keys, and rules are added
greedily under a residual objective, the sparse evaluation of ProposalDistribution's
negative phase.

Fixes over the first version, following the diagnosis in results_step1.md:
  * candidate bodies come from the full relation alphabet, pruned by the necessary condition
    that body domain/range intersect the head's (contraction-order argument, cheapest form)
  * bodies may carry three premises, extended from the surviving two-premise bodies
  * rule scores aggregate by maximum firing confidence instead of a sum
  * the residual stopping rule is a parameter (minSupport) rather than a constant

Usage:
    python boost.py --selftest
    python boost.py --depth 3 --rules 20 --minSupport 2 --aggregation max
"""

import argparse
import os
import time

import numpy as np
from scipy import sparse

import kg_rules as K

HERE = os.path.dirname(os.path.abspath(__file__))


def encode(rows, cols, n):
    return rows.astype(np.int64) * n + cols.astype(np.int64)


def sorted_keys(matrix, n):
    """Sorted encoded keys of the nonzeros. Matmul returns non-canonical index order."""
    counts = np.diff(matrix.indptr)
    rows = np.repeat(np.arange(matrix.shape[0], dtype=np.int64), counts)
    return np.sort(encode(rows, matrix.indices, n))


def build_fact_index(rows, cols, rIdx, n):
    keys = encode(rows, cols, n)
    order = np.argsort(keys, kind="stable")
    return keys[order], rIdx[order]


def domains_ranges(cores):
    dom, rang = [], []
    for core in cores:
        rows = np.repeat(np.arange(core.shape[0], dtype=np.int64), np.diff(core.indptr))
        dom.append(np.unique(rows))
        rang.append(np.unique(core.indices.astype(np.int64)))
    return dom, rang


def overlaps(target, others):
    """Boolean vector: does `target` intersect each array in `others`."""
    out = np.zeros(len(others), dtype=bool)
    if target.size == 0:
        return out
    for i, other in enumerate(others):
        if other.size == 0:
            continue
        pos = np.minimum(np.searchsorted(other, target), other.size - 1)
        out[i] = bool((other[pos] == target).any())
    return out


def stride(items, limit):
    if len(items) <= limit:
        return items
    step = len(items) / float(limit)
    return [items[int(i * step)] for i in range(limit)]


_POLY_LEFT = None
_POLY_RIGHT = None
_SPARSE_LEFT = None
_SPARSE_RIGHT = None
_FEATURES = None
_NEGATIVE = []
_ACCEPTANCE = "info_gain"


def load_features(path, entities):
    """One numeric feature per entity, zero where absent."""
    index = {e: i for i, e in enumerate(entities)}
    values = np.zeros(len(entities), dtype=np.float64)
    seen = np.zeros(len(entities), dtype=bool)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or parts[0] not in index:
                continue
            values[index[parts[0]]] = float(parts[1])
            seen[index[parts[0]]] = True
    return values, seen


def comparison_mask(matrix, features, operator):
    """Keep the entries of a relation whose feature comparison holds.

    The comparison factor is dense in general, but a body only ever needs it where a relational atom
    has an edge, so the operator reduces to a mask over the existing nonzeros. Cost is the nonzero
    count, not the square of the entity count.
    """
    coo = matrix.tocoo()
    if operator == ">":
        keep = features[coo.row] > features[coo.col]
    elif operator == "<":
        keep = features[coo.row] < features[coo.col]
    else:
        raise ValueError("unknown comparison " + operator)
    return sparse.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=matrix.shape)


def build_sparse_cores(cores, n):
    """One tnreason sparse core per atom, in both orientations, contracted by the compiled kernel."""
    from tnreason import engine
    left, right = [], []
    for c in cores:
        left.append(engine.get_core("ScipySparseCore")(values=c, colors=["X", "Z"], shape=[n, n],
                                                       name="left"))
        right.append(engine.get_core("ScipySparseCore")(values=c, colors=["Z", "Y"], shape=[n, n],
                                                        name="right"))
    return left, right


def build_poly_cores(cores, n):
    """One PolynomialCore per atom, in two orientations so any body can be read as a chain.

    The monomial keys must be the colours the core declares, otherwise the shared variable never
    matches and every contraction is empty.
    """
    from tnreason import engine
    left, right = [], []
    for c in cores:
        xs, ys = c.nonzero()
        left.append(engine.get_core("PolynomialCore")(
            values=[(1.0, {"X": int(x), "Z": int(y)}) for x, y in zip(xs, ys)],
            colors=["X", "Z"], shape=[n, n], name="left"))
        right.append(engine.get_core("PolynomialCore")(
            values=[(1.0, {"Z": int(x), "Y": int(y)}) for x, y in zip(xs, ys)],
            colors=["Z", "Y"], shape=[n, n], name="right"))
    return left, right


def sparse_chain(cores, body):
    """Price a two-literal body through tnreason's compiled sparse contractor."""
    from tnreason import engine
    a, b = body
    res = engine.contract(coreDict={"l": _SPARSE_LEFT[a], "r": _SPARSE_RIGHT[b]},
                          openColors=["X", "Y"], contractionMethod="ScipySparse")
    return res.values.tocsr()


def poly_chain(cores, body):
    """Price a two-literal body through tnreason's polynomial contractor, returned as a sparse matrix."""
    from tnreason import engine
    a, b = body
    res = engine.contract(coreDict={"l": _POLY_LEFT[a], "r": _POLY_RIGHT[b]},
                          openColors=["X", "Y"], contractionMethod="CorewiseContractor")
    rows, cols = [], []
    for value, position in res.values:
        if value != 0 and "X" in position and "Y" in position:
            rows.append(position["X"])
            cols.append(position["Y"])
    n = _POLY_LEFT[a].shape[0]
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))


def diagonal(keep):
    """Sparse diagonal of a 0/1 vector: masks rows or columns without dense broadcasting."""
    n = keep.size
    return sparse.csr_matrix((keep.astype(np.float32), (np.arange(n), np.arange(n))), shape=(n, n))


def chain(cores, body):
    """Score matrix over (X,Y) for a body: a path, or a fork joined off that path.

    A body is either a tuple of atom indices, meaning a chain X -> ... -> Y, or
    ("fz", a, b) meaning r_a(X,Z) and r_b(Y,Z) with Z shared off the path, or
    ("fx", a, b) meaning r_a(Z,X) and r_b(Z,Y).
    """
    if _SPARSE_LEFT is not None and not isinstance(body[0], str) and len(body) == 2:
        return sparse_chain(cores, body)
    if _POLY_LEFT is not None and not isinstance(body[0], str) and len(body) == 2:
        return poly_chain(cores, body)
    if isinstance(body[0], str) and body[0] == "neg":
        # r_a(X,Z) & r_b(Z,Y) & not r_c(X,Y): the negated atom is evaluated only where the
        # rest of the body has support, so the complement is never formed.
        base = chain(cores, tuple(body[1:3]))
        blocked = cores[body[3]]
        return (base - base.multiply(blocked)).tocsr()
    if isinstance(body[0], str) and body[0] in ("cmp>", "cmp<"):
        operator, a, b = body[0][3:], body[1], body[2]
        return (comparison_mask(cores[a], _FEATURES, operator) @ cores[b]).tocsr()
    if isinstance(body[0], str):
        mode, atoms = body[0], body[1:]
        if mode == "fz":
            return (cores[atoms[0]] @ cores[atoms[1]].T).tocsr()
        if mode == "fx":
            return (cores[atoms[0]].T @ cores[atoms[1]]).tocsr()
        if mode in ("tx", "ty", "tz"):
            # three-literal tree: a chain r(X,Z), s(Z,Y) with a third premise t hung off
            # X (tx), Y (ty) or the intermediate Z (tz); the hung variable is existential
            a, b, t = atoms
            path = (cores[a] @ cores[b]).tocsr()
            if mode == "tx":
                keep = np.asarray((cores[t].getnnz(axis=1) > 0)).ravel()
                return (diagonal(keep) @ path).tocsr()
            if mode == "ty":
                keep = np.asarray((cores[t].getnnz(axis=0) > 0)).ravel()
                return (path @ diagonal(keep)).tocsr()
            keep = np.asarray((cores[t].getnnz(axis=1) > 0)).ravel()
            return (cores[a] @ diagonal(keep) @ cores[b]).tocsr()
        raise ValueError("unknown topology " + mode)
    out = cores[body[0]]
    for k in body[1:]:
        out = out @ cores[k]
    return out


def price_bodies(cores, n, factKeys, factRels, nRel, bodies):
    """support[head, body] = head facts (x,y) for which the body holds for some intermediates."""
    support = np.zeros((nRel, len(bodies)), dtype=np.float32)
    bodyNnz = np.zeros((nRel, len(bodies)), dtype=np.float32)
    paths = {}
    t0 = time.time()
    for k, body in enumerate(bodies):
        path = chain(cores, body)
        if path.nnz == 0:
            continue
        paths[k] = path
        pkeys = sorted_keys(path, n)
        pos = np.minimum(np.searchsorted(pkeys, factKeys), pkeys.size - 1)
        hit = pkeys[pos] == factKeys
        support[:, k] = np.bincount(factRels[hit], minlength=nRel)
        bodyNnz[:, k] = path.nnz
    print("  priced {} bodies in {:.1f}s".format(len(bodies), time.time() - t0), flush=True)
    return support, bodyNnz, paths


def learn_rules(entities, relations, rows, cols, rIdx, depth, ruleCount, minSupport, maxBodies=4000,
                inverse=False, constants=False, topology="chain"):
    n, nRel = len(entities), len(relations)
    headCores = K.relation_cores(rows, cols, rIdx, n, nRel)
    cores = headCores + [c.T.tocsr() for c in headCores] if inverse else headCores
    nAtom = len(cores)
    factKeys, factRels = build_fact_index(rows, cols, rIdx, n)
    dom, rang = domains_ranges(cores)

    domOverlap = np.array([overlaps(dom[h], dom) for h in range(nRel)])
    rangOverlap = np.array([overlaps(rang[h], rang) for h in range(nRel)])
    joint = np.array([overlaps(rang[a], dom) for a in range(nAtom)])

    bodies = []
    # trees hang off a two-literal chain, so the chain base is always enumerated
    if topology in ("chain", "both", "tree", "all"):
        bodies += [(a, b) for a in range(nAtom) for b in range(nAtom) if joint[a, b]]
    if topology in ("fork", "both", "all"):
        # fork: the shared intermediate is reached from both head variables, so the
        # pairing constraint is range-range (fz) or domain-domain (fx), not range-domain
        rangSelf = np.array([overlaps(rang[a], rang) for a in range(nAtom)])
        domSelf = np.array([overlaps(dom[a], dom) for a in range(nAtom)])
        bodies += [("fz", a, b) for a in range(nAtom) for b in range(nAtom) if rangSelf[a, b]]
        bodies += [("fx", a, b) for a in range(nAtom) for b in range(nAtom) if domSelf[a, b]]
    if _FEATURES is not None:
        # a comparison between the head variable and the intermediate, on top of a two-literal chain
        cmpBodies = []
        for a in range(nAtom):
            for b in np.nonzero(joint[a])[0]:
                for operator in (">", "<"):
                    cmpBodies.append(("cmp" + operator, a, int(b)))
        bodies += cmpBodies
        print("{} comparison bodies added (feature range {:.3g} to {:.3g})".format(
            len(cmpBodies), float(_FEATURES.min()), float(_FEATURES.max())), flush=True)
    if _NEGATIVE:
        # negating a sparse relation is nearly vacuous, so only dense relations are candidates
        negBodies = []
        for a in range(nAtom):
            for b in np.nonzero(joint[a])[0]:
                for c in _NEGATIVE:
                    negBodies.append(("neg", a, int(b), c))
        bodies += negBodies
        print("{} negation bodies added over {} dense relations".format(
            len(negBodies), len(_NEGATIVE)), flush=True)
    print("{} two-premise bodies pass pruning over {} atoms, topology={}".format(
        len(bodies), nAtom, topology), flush=True)
    # ponytail: uniform stride keeps every relation index range represented; switch to
    # support-based pruning only if the stride measurably loses rules
    bodies = stride(bodies, maxBodies)
    support, bodyNnz, paths = price_bodies(cores, n, factKeys, factRels, nRel, bodies)
    allBodies = list(bodies)

    if depth >= 3:
        extensions = set()
        for h in range(nRel):
            viable = [k for k, (a, b) in enumerate(allBodies) if domOverlap[h, a] and rangOverlap[h, b]]
            viable = sorted(viable, key=lambda k: -support[h, k])[:5]
            for k in viable:
                if support[h, k] < minSupport:
                    continue
                body = allBodies[k]
                if isinstance(body[0], str):
                    continue                      # extensions are defined for chains only
                a, b = body
                for c in range(nAtom):
                    if joint[b, c]:
                        extensions.add((a, b, c))
        extensions = stride(sorted(extensions), maxBodies)
        print("{} three-premise bodies after extension and pruning".format(len(extensions)), flush=True)
        if extensions:
            s3, n3, p3 = price_bodies(cores, n, factKeys, factRels, nRel, extensions)
            start = len(allBodies)
            allBodies.extend(extensions)
            for k in p3:                      # bodies with empty paths keep their column
                paths[start + k] = p3[k]
            support = np.hstack([support, s3])
            bodyNnz = np.hstack([bodyNnz, n3])

    if topology in ("tree", "all"):
        domSelf3 = np.array([overlaps(dom[a], dom) for a in range(nAtom)])
        rangSelf3 = np.array([overlaps(rang[a], rang) for a in range(nAtom)])
        trees = set()
        for h in range(nRel):
            viable = [k for k, body in enumerate(allBodies[:len(bodies)]) if not isinstance(body[0], str)
                      and domOverlap[h, body[0]] and rangOverlap[h, body[1]]]
            viable = sorted(viable, key=lambda k: -support[h, k])[:5]
            for k in viable:
                if support[h, k] < minSupport:
                    continue
                a, b = allBodies[k]
                for t in range(nAtom):
                    if domSelf3[t, a]:
                        trees.add(("tx", a, b, t))
                    if rangSelf3[t, b]:
                        trees.add(("ty", a, b, t))
                    if domSelf3[t, b]:
                        trees.add(("tz", a, b, t))
        trees = stride(sorted(trees), maxBodies)
        print("{} three-literal trees after pruning".format(len(trees)), flush=True)
        if trees:
            s3, n3, p3 = price_bodies(cores, n, factKeys, factRels, nRel, trees)
            start = len(allBodies)
            allBodies.extend(trees)
            for k in p3:
                paths[start + k] = p3[k]
            support = np.hstack([support, s3])
            bodyNnz = np.hstack([bodyNnz, n3])

    headKeys = [np.sort(encode(rows[rIdx == h], cols[rIdx == h], n)) for h in range(nRel)]
    candKeys = {k: sorted_keys(p, n) for k, p in paths.items()}

    headSizes = [int(headCores[h].nnz) for h in range(nRel)]
    ruleSets, shortlistSize = {}, max(60, 4 * ruleCount)
    for h in range(nRel):
        ruleSets[h] = []
        if support[h].max() < minSupport:
            continue
        keys = headKeys[h]
        hitsOf = {}
        for k in np.argsort(-support[h])[:shortlistSize]:
            k = int(k)
            if support[h, k] < minSupport or k not in candKeys:
                continue
            pos = np.minimum(np.searchsorted(candKeys[k], keys), candKeys[k].size - 1)
            hitsOf[k] = np.nonzero(candKeys[k][pos] == keys)[0]
        covered = np.zeros(keys.size, dtype=bool)
        baseRate = headSizes[h] / float(n * n)
        for _ in range(ruleCount):
            best = None
            for k, hits in hitsOf.items():
                if any(rule["cand"] == k for rule in ruleSets[h]):
                    continue
                residual = int((~covered[hits]).sum())
                if residual < minSupport:
                    continue
                precision = residual / max(float(bodyNnz[h, k]), 1.0)
                gain = (info_gain(residual, precision, baseRate) if _ACCEPTANCE == "info_gain"
                        else likelihood_gain(residual, float(support[h, k]), float(bodyNnz[h, k]),
                                             baseRate))
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, k, residual, precision)
            if best is None:
                break
            gain, k, residual, precision = best
            covered[hitsOf[k]] = True
            ruleSets[h].append({"cand": k, "body": allBodies[k], "support": float(support[h, k]),
                                "confidence": float(support[h, k] / max(float(bodyNnz[h, k]), 1.0)),
                                "residual": residual, "gain": float(gain)})
        if ruleSets[h]:
            top = ruleSets[h][0]
            print("{:<42} rules={} conf={:.3f} depth={} body={} support={:g}".format(
                relations[h][:42], len(ruleSets[h]), top["confidence"],
                len(top["body"]), ruleText(relations, top["body"], nRel), top["support"]),
                flush=True)
    if constants:
        constSets = constant_rules(cores, headCores, n, nRel, nAtom, dom, rang, relations,
                                   perHead=ruleCount, minSupport=minSupport)
        for h, extra in constSets.items():
            ruleSets.setdefault(h, []).extend(extra)
    total = sum(len(v) for v in ruleSets.values())
    print("learned {} rules over {} head relations".format(total, nRel), flush=True)
    return cores, ruleSets, allBodies, n, total


def constant_rules(cores, headCores, n, nRel, nAtom, dom, rang, relations,
                   perHead=6, minSupport=5, atomCap=40, maxBroadcast=2_000_000):
    """One-premise rules whose premise carries an entity constant, as in AnyBURL.

    Four families, distinguished by whether the constant binds the first or second argument
    of the premise and by which head variable the premise constrains: r(X,c), r(c,X),
    s(Y,c), s(c,Y). Each family is priced for all constants at once by one sparse
    matrix-vector product per head relation and atom.

    ponytail: one premise only; a constant combined with a chain premise needs the same
    matvec on a shifted indicator and is deferred until one-premise constants stop paying.
    """
    ruleSets = {}
    priced = 0
    rangDom = np.array([overlaps(rang[h], dom) for h in range(nRel)])
    domRang = np.array([overlaps(dom[h], rang) for h in range(nRel)])
    for h in range(nRel):
        H = headCores[h].tocoo()
        if H.nnz == 0:
            continue
        uHead = np.zeros(n)
        uHead[H.row] = 1.0
        uTail = np.zeros(n)
        uTail[H.col] = 1.0
        baseRate = H.nnz / float(n * n)
        candidates = []
        atoms = [a for a in range(nAtom) if rangDom[h, a] or domRang[h, a]][:atomCap]
        for a in atoms:
            A = cores[a]
            colSum = np.asarray(A.sum(axis=0)).ravel()
            rowSum = np.asarray(A.sum(axis=1)).ravel()
            if rangDom[h, a]:                             # constant in second argument
                for side, u in (("X", uHead), ("Y", uTail)):
                    hits = np.asarray(u @ A).ravel()
                    candidates.extend(priceConstants(hits, colSum, a, side, "arg2", baseRate, minSupport))
            if domRang[h, a]:                             # constant in first argument
                for side, u in (("X", uHead), ("Y", uTail)):
                    hits = np.asarray(A @ u).ravel()
                    candidates.extend(priceConstants(hits, rowSum, a, side, "arg1", baseRate, minSupport))
            priced += 1
        if not candidates:
            continue
        candidates.sort(key=lambda c: -c[0])
        accepted, seen = [], set()
        for gain, hits, precision, a, side, where, constIdx in candidates:
            key = (a, side, where, constIdx)
            if key in seen:
                continue
            seen.add(key)
            vec = np.zeros(n)
            if where == "arg2":
                vec = np.asarray(cores[a][:, constIdx].todense()).ravel()
            else:
                vec = np.asarray(cores[a][constIdx, :].todense()).ravel()
            nnzVec = int(np.count_nonzero(vec))
            if nnzVec == 0 or nnzVec * n > maxBroadcast:
                continue
            rowsIdx = np.nonzero(vec)[0]
            data = np.repeat(vec[rowsIdx], n)
            if side == "X":
                mat = sparse.csr_matrix((data, (np.repeat(rowsIdx, n), np.tile(np.arange(n), nnzVec))),
                                        shape=(n, n))
            else:
                mat = sparse.csr_matrix((data, (np.tile(np.arange(n), nnzVec), np.repeat(rowsIdx, n))),
                                        shape=(n, n))
            accepted.append({"matrix": mat, "confidence": float(precision), "support": float(hits),
                             "constant": (a, side, where, constIdx), "gain": float(gain),
                             "dir": "head" if side == "X" else "tail"})
            if len(accepted) >= perHead:
                break
        if accepted:
            ruleSets[h] = accepted
    print("priced constants over {} head-relation and atom pairs, kept {} constant rules".format(
        priced, sum(len(v) for v in ruleSets.values())), flush=True)
    return ruleSets


def priceConstants(hits, coverage, atom, side, where, baseRate, minSupport):
    out = []
    mask = hits >= minSupport
    if not mask.any():
        return out
    idx = np.nonzero(mask)[0]
    for c in idx:
        cov = float(coverage[c])
        if cov <= 0:
            continue
        precision = hits[c] / cov
        gain = info_gain(float(hits[c]), precision, baseRate)
        if gain > 0:
            out.append((gain, float(hits[c]), precision, atom, side, where, int(c)))
    return out


def info_gain(mass, confidence, prior):
    """Residual mass times the log-odds improvement over the base rate.

    This is the negative phase entering as the prior: a rule is worth adding only if its
    precision on the uncovered mass beats the head relation's base rate.
    """
    c = min(max(confidence, 1e-12), 1 - 1e-12)
    p = min(max(prior, 1e-12), 1 - 1e-12)
    return mass * (c * np.log(c / p) + (1 - c) * np.log((1 - c) / (1 - p)))


def likelihood_gain(residual, support, bodyNnz, baseRate):
    """Positive minus negative phase of the likelihood gradient, on the uncovered mass.

    Adding a body to the model changes the log-likelihood by the mass it places on facts that are not
    yet explained, minus the mass it places where the current model expects facts and finds none. The
    current model is a base-rate marginal, which is the first term of a contrastive-divergence step.
    """
    positive = float(residual)
    negative = max(float(bodyNnz - support), 0.0) * min(max(baseRate, 0.0), 1.0)
    return positive - negative


def exportRules(ruleSets, relations, nRel, path):
    """Write the learned rule set in AnyBURL's rule-file format for external scoring."""
    nBodies, skipped = 0, 0
    with open(path, "w", encoding="utf-8") as f:
        for h, rules in ruleSets.items():
            for rule in rules:
                if "body" not in rule or isinstance(rule["body"][0], str):
                    skipped += 1          # constant and fork rules have no chain form
                    continue
                atoms, prev = [], "X"
                body = rule["body"]
                for i, atom in enumerate(body):
                    rel = atom if atom < nRel else atom - nRel
                    inverted = atom >= nRel
                    nxt = "Y" if i == len(body) - 1 else "ABC"[i]
                    u, v = (nxt, prev) if inverted else (prev, nxt)
                    atoms.append("{}({},{})".format(relations[rel], u, v))
                    prev = nxt
                f.write("{}\t{}\t{}\t{}(X,Y) <= {}\n".format(
                    int(rule["support"]), len(body), rule["confidence"], relations[h], ", ".join(atoms)))
                nBodies += 1
    print("exported {} rules to {} ({} constant rules not representable)".format(
        nBodies, path, skipped), flush=True)


def ruleText(relations, body, nRel):
    if isinstance(body[0], str) and body[0] == "neg":
        return "{}(X,Z) & {}(Z,Y) & not {}(X,Y)".format(
            atomName(relations, body[1], nRel), atomName(relations, body[2], nRel),
            atomName(relations, body[3], nRel))
    if isinstance(body[0], str) and body[0] in ("cmp>", "cmp<"):
        operator = ">" if body[0] == "cmp>" else "<"
        return "{}(X,Z) & feature(X) {} feature(Z) & {}(Z,Y)".format(
            atomName(relations, body[1], nRel), operator, atomName(relations, body[2], nRel))
    if isinstance(body[0], str):
        return " & ".join(atomName(relations, a, nRel) for a in body[1:])
    return ".".join(atomName(relations, a, nRel) for a in body)


def atomName(relations, atom, nRel):
    return relations[atom] if atom < nRel else "inv(" + relations[atom - nRel] + ")"


def read_triples(path, entIndex, relIndex):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            if parts[0] in entIndex and parts[2] in entIndex and parts[1] in relIndex:
                out.append((entIndex[parts[0]], relIndex[parts[1]], entIndex[parts[2]]))
    return out


def evaluate(cores, ruleSets, n, relations, entIndex, relIndex, trainIdx, extraPaths, testPath,
             maxTest=None, priorOnly=False, aggregation="max", direction="both", dumpPath=None):
    known, knownHeads = {}, {}
    for h, r, t in trainIdx:
        known.setdefault((h, r), set()).add(t)
        knownHeads.setdefault((r, t), set()).add(h)
    for path in extraPaths:
        for h, r, t in read_triples(path, entIndex, relIndex):
            known.setdefault((h, r), set()).add(t)
            knownHeads.setdefault((r, t), set()).add(h)
    test = read_triples(testPath, entIndex, relIndex)
    if maxTest:
        test = test[:maxTest]
    for h, r, t in test:
        known.setdefault((h, r), set()).add(t)
        knownHeads.setdefault((r, t), set()).add(h)

    byHead = {}
    for h, r, t in test:
        byHead.setdefault(r, []).append((h, t))

    tailCounts = np.zeros((len(relations), n), dtype=np.float32)
    headCounts = np.zeros((len(relations), n), dtype=np.float32)
    np.add.at(tailCounts, (np.array([r for _, r, _ in trainIdx], dtype=np.int64),
                           np.array([t for _, _, t in trainIdx], dtype=np.int64)), 1.0)
    np.add.at(headCounts, (np.array([r for _, r, _ in trainIdx], dtype=np.int64),
                           np.array([h for h, _, _ in trainIdx], dtype=np.int64)), 1.0)

    ranks, positives, headRanks, headPositives, t0 = [], [], [], [], time.time()
    unruled = sum(1 for hIdx in byHead if not ruleSets.get(hIdx))
    for hIdx in sorted(byHead):
        rules = ruleSets.get(hIdx, [])
        score, scoreHead = None, None
        for rule in ([] if priorOnly else rules):
            path = rule["matrix"] if "matrix" in rule else chain(cores, rule["body"])
            term = (path > 0).astype(np.float32) * rule["confidence"]
            # a rule that leaves the ranked variable unconstrained scores every candidate
            # alike; under max aggregation it would beat every discriminative rule
            dirn = rule.get("dir", "both")
            if dirn != "head":
                if score is None:
                    score = term
                elif aggregation == "max":
                    score = score.maximum(term)
                else:
                    score = score + term
            if dirn != "tail":
                if scoreHead is None:
                    scoreHead = term
                elif aggregation == "max":
                    scoreHead = scoreHead.maximum(term)
                else:
                    scoreHead = scoreHead + term
        pairs = byHead[hIdx]
        targetRows = np.array(sorted({x for x, _ in pairs}))
        sub = np.zeros((targetRows.size, n), dtype=np.float64) if score is None \
            else score[targetRows, :].toarray().astype(np.float64)
        # ties broken by the head relation's empirical tail marginal, then by entity index:
        # score 0 means no rule fired, which is not evidence of a wrong answer
        prior = tailCounts[hIdx] / max(float(tailCounts[hIdx].max()), 1.0)
        sub = sub + 1e-6 * prior[None, :] - 1e-9 * np.arange(n, dtype=np.float64)[None, :]
        rowOf = {int(x): i for i, x in enumerate(targetRows)}
        for x, y in pairs:
            row = sub[rowOf[x]]
            trueTails = np.array([t for t in known[(x, hIdx)] if t != y], dtype=np.int64)
            target = row[y]
            if trueTails.size:
                row = row.copy()
                row[trueTails] = -np.inf
            ranks.append(1 + int((row > target).sum()))
            positives.append(int((row > 0).sum()))

        if direction == "both":
            # head direction: rank x among all heads for the same (relation, tail)
            targetTails = np.array(sorted({y for _, y in pairs}))
            colOf = {int(y): i for i, y in enumerate(targetTails)}
            priorH = headCounts[hIdx] / max(float(headCounts[hIdx].max()), 1.0)
            headScore = scoreHead if scoreHead is not None else score
            for start in range(0, targetTails.size, 512):
                chunk = targetTails[start:start + 512]
                block = np.zeros((chunk.size, n), dtype=np.float64) if headScore is None \
                    else np.asarray(headScore[:, chunk].todense(), dtype=np.float64).T
                block += 1e-6 * priorH[None, :] - 1e-9 * np.arange(n, dtype=np.float64)[None, :]
                for x, y in pairs:
                    i = colOf[int(y)] - start
                    if i < 0 or i >= chunk.size:
                        continue
                    row = block[i]
                    trueHeads = np.array([hh for hh in knownHeads[(hIdx, y)] if hh != x], dtype=np.int64)
                    target = row[x]
                    if trueHeads.size:
                        row = row.copy()
                        row[trueHeads] = -np.inf
                    headRanks.append(1 + int((row > target).sum()))
                    headPositives.append(int((row > 0).sum()))
    ranks = np.array(ranks, dtype=np.float64)
    positives = np.array(positives, dtype=np.float64)
    if dumpPath:
        np.savez(dumpPath, tail=ranks, head=np.array(headRanks, dtype=np.float64))
        print("dumped {} tail and {} head ranks to {}".format(ranks.size, len(headRanks), dumpPath),
              flush=True)
    if ranks.size == 0:
        print("no test triple covered by any rule", flush=True)
        return None
    headRanks = np.array(headRanks, dtype=np.float64)
    headPositives = np.array(headPositives, dtype=np.float64)
    print("\nfiltered evaluation on {} test triples, {} head relations without rules ({:.1f}s), "
          "aggregation={}, direction={}".format(ranks.size, unruled, time.time() - t0, aggregation, direction),
          flush=True)
    print("mean scored candidates per query: {:.2f} of {}".format(positives.mean(), n), flush=True)
    print("tail  MRR {:.4f}  Hits@1 {:.4f}  Hits@3 {:.4f}  Hits@10 {:.4f}".format(
        float((1.0 / ranks).mean()), float((ranks <= 1).mean()),
        float((ranks <= 3).mean()), float((ranks <= 10).mean())), flush=True)
    if headRanks.size:
        print("mean scored candidates per query (head): {:.2f} of {}".format(headPositives.mean(), n), flush=True)
        print("head  MRR {:.4f}  Hits@1 {:.4f}  Hits@3 {:.4f}  Hits@10 {:.4f}".format(
            float((1.0 / headRanks).mean()), float((headRanks <= 1).mean()),
            float((headRanks <= 3).mean()), float((headRanks <= 10).mean())), flush=True)
        allRanks = np.concatenate([ranks, headRanks])
        print("both  MRR {:.4f}  Hits@1 {:.4f}  Hits@3 {:.4f}  Hits@10 {:.4f}".format(
            float((1.0 / allRanks).mean()), float((allRanks <= 1).mean()),
            float((allRanks <= 3).mean()), float((allRanks <= 10).mean())), flush=True)
        return float((1.0 / allRanks).mean())
    return float((1.0 / ranks).mean())


def forkSelftest():
    """h(x,y) holds exactly when some z has a(x,z) and b(y,z): the fork must recover it."""
    n, nRel = 150, 3
    rng = np.random.default_rng(11)
    k = 3
    A = np.zeros((n, n), dtype=bool)
    B = np.zeros((n, n), dtype=bool)
    A[np.arange(n)[:, None], rng.integers(0, n, size=(n, k))] = True
    B[np.arange(n)[:, None], rng.integers(0, n, size=(n, k))] = True
    H = (A.astype(np.int32) @ B.astype(np.int32).T) > 0
    rows, cols, rels = [], [], []
    for mat, rel in ((A, 0), (B, 1), (H, 2)):
        xs, ys = np.nonzero(mat)
        rows.extend(xs.tolist()); cols.extend(ys.tolist()); rels.extend([rel] * len(xs))
    rows, cols, rels = np.array(rows), np.array(cols), np.array(rels)
    cores = K.relation_cores(rows, cols, rels, n, nRel)
    body = ("fz", 0, 1)
    path = chain(cores, body)
    assert path.nnz == H.sum(), "fork scores {} pairs, planted {} ".format(path.nnz, H.sum())
    chainPath = chain(cores, (0, 1))
    assert chainPath.nnz != H.sum(), "planted fork must not be expressible as a chain"
    print("fork selftest passed: fork recovers {} pairs, chain gives {}".format(
        path.nnz, chainPath.nnz), flush=True)


def treeSelftest():
    """h(x,y) holds when a(x,z) ^ b(z,y) and x has any c(x,w): the tx tree must recover it."""
    n, nRel = 150, 4
    rng = np.random.default_rng(5)
    k = 3
    A = np.zeros((n, n), dtype=bool)
    B = np.zeros((n, n), dtype=bool)
    C = np.zeros((n, n), dtype=bool)
    A[np.arange(n)[:, None], rng.integers(0, n, size=(n, k))] = True
    B[np.arange(n)[:, None], rng.integers(0, n, size=(n, k))] = True
    sparseRows = np.where(np.arange(n) % 3 == 0)[0]          # only a third of x have a c edge
    C[sparseRows[:, None], rng.integers(0, n, size=(sparseRows.size, k))] = True
    H = ((A.astype(np.int32) @ B.astype(np.int32)) > 0) & (C.any(axis=1)[:, None])
    rows, cols, rels = [], [], []
    for mat, rel in ((A, 0), (B, 1), (C, 2), (H, 3)):
        xs, ys = np.nonzero(mat)
        rows.extend(xs.tolist()); cols.extend(ys.tolist()); rels.extend([rel] * len(xs))
    cores = K.relation_cores(np.array(rows), np.array(cols), np.array(rels), n, nRel)
    tree = chain(cores, ("tx", 0, 1, 2))
    assert tree.nnz == H.sum(), "tx tree scores {} pairs, planted {}".format(tree.nnz, H.sum())
    plain = chain(cores, (0, 1))
    assert plain.nnz != H.sum(), "planted tree must not equal the bare chain"
    tz = chain(cores, ("tz", 0, 1, 2))
    assert tz.shape == (n, n) and tz.nnz > 0, "tz tree did not score"
    print("tree selftest passed: tx recovers {} pairs, bare chain {}".format(tree.nnz, plain.nnz), flush=True)


def constSelftest():
    """A rule h(X,Y) <= r(X,c) planted exactly must return precision 1.0."""
    n, nRel = 200, 2
    rIdx, cIdx = 0, 1
    const = 7
    rowsR = np.arange(50)
    colsR = np.full(50, const)
    rowsH = np.arange(50)
    colsH = np.arange(50) % 4
    rows = np.concatenate([rowsR, rowsH])
    cols = np.concatenate([colsR, colsH])
    rels = np.concatenate([np.full(50, rIdx), np.full(50, cIdx)])
    cores = K.relation_cores(rows, cols, rels, n, nRel)
    dom, rang = domains_ranges(cores)
    relations = ["r", "h"]
    sets = constant_rules(cores, cores, n, nRel, nRel, dom, rang, relations, perHead=4, minSupport=5)
    assert cIdx in sets, "no constant rule found for the head relation"
    bestConf = max(rule["confidence"] for rule in sets[cIdx])
    assert abs(bestConf - 1.0) < 1e-9, "planted constant rule has precision {}".format(bestConf)
    sides = {rule["constant"][1] for rule in sets[cIdx]}
    assert "X" in sides, "constant rule does not constrain the head entity"
    print("constant selftest passed: {} rules, best precision 1.000, sides {}".format(
        len(sets[cIdx]), sorted(sides)), flush=True)


def selftest():
    """Planted composition c = a o b must be recovered with confidence one."""
    rng = np.random.default_rng(1)
    n = 200

    def functional(k=2):
        m = np.zeros((n, n), dtype=bool)
        m[np.arange(n)[:, None], rng.integers(0, n, size=(n, k))] = True
        return m

    a, b = functional(), functional()
    c = (a.astype(np.int32) @ b.astype(np.int32)) > 0
    rows, cols, rIdx = [], [], []
    for mat, rel in ((a, 0), (b, 1), (c, 2)):
        xs, ys = np.nonzero(mat)
        rows.extend(xs.tolist())
        cols.extend(ys.tolist())
        rIdx.extend([rel] * len(xs))
    rows, cols, rIdx = np.array(rows), np.array(cols), np.array(rIdx)
    entities = ["e" + str(i) for i in range(n)]
    relations = ["a", "b", "c"]
    cores, ruleSets, bodies, _, _ = learn_rules(entities, relations, rows, cols, rIdx,
                                                depth=2, ruleCount=2, minSupport=3)
    assert ruleSets[2], "no rule learned for the planted head relation"
    body = ruleSets[2][0]["body"]
    print("selftest learned body:", [relations[r] for r in body], flush=True)
    assert body == (0, 1), "planted body not recovered, got {}".format(body)
    assert abs(ruleSets[2][0]["confidence"] - 1.0) < 1e-6, "confidence of an exact composition must be one"
    print("selftest passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--dir", default=os.path.join(HERE, "data", "fb15k237"))
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--rules", type=int, default=20)
    parser.add_argument("--minSupport", type=int, default=2)
    parser.add_argument("--maxBodies", type=int, default=4000)
    parser.add_argument("--aggregation", default="max", choices=["max", "sum"])
    parser.add_argument("--maxTest", type=int, default=None)
    parser.add_argument("--priorOnly", action="store_true")
    parser.add_argument("--inverse", action="store_true")
    parser.add_argument("--export", default=None)
    parser.add_argument("--constants", action="store_true")
    parser.add_argument("--topology", default="chain", choices=["chain", "fork", "both", "tree", "all"])
    parser.add_argument("--dumpRanks", default=None)
    parser.add_argument("--pricing", default="sparse", choices=["sparse", "tnreason", "tnreason-sparse"])
    parser.add_argument("--features", default=None)
    parser.add_argument("--acceptance", default="info_gain", choices=["info_gain", "likelihood"])
    parser.add_argument("--negation", type=float, default=None,
                        help="negate relations whose density exceeds this value")
    parser.add_argument("--constselftest", action="store_true")
    parser.add_argument("--forkselftest", action="store_true")
    parser.add_argument("--treeselftest", action="store_true")
    args = parser.parse_args()
    if args.treeselftest:
        treeSelftest()
    elif args.forkselftest:
        forkSelftest()
    elif args.constselftest:
        constSelftest()
    elif args.selftest:
        selftest()
    else:
        trainPath = os.path.join(args.dir, "train.txt")
        entities, relations, rows, cols, rIdx = K.load_triples(trainPath)
        if args.features:
            import sys as _sys
            values, _seen = load_features(args.features, entities)
            _sys.modules[__name__]._FEATURES = values
        import sys as _sys
        _sys.modules[__name__]._ACCEPTANCE = args.acceptance
        if args.negation is not None:
            forward = K.relation_cores(rows, cols, rIdx, len(entities), len(relations))
            densities = np.asarray([m.nnz for m in forward], dtype=np.float64) / \
                float(len(entities) ** 2)
            dense = [i for i, d in enumerate(densities) if d >= args.negation]
            _sys.modules[__name__]._NEGATIVE = dense
            print("{} of {} relations dense enough to negate (density >= {}): {}".format(
                len(dense), len(densities), args.negation,
                ", ".join(relations[i] for i in dense[:6])), flush=True)
        entIndex = {e: i for i, e in enumerate(entities)}
        relIndex = {r: i for i, r in enumerate(relations)}
        if args.pricing == "tnreason-sparse":
            import sys as _sys
            module = _sys.modules[__name__]
            headCores = K.relation_cores(rows, cols, rIdx, len(entities), len(relations))
            allCores = headCores + [c.T.tocsr() for c in headCores] if args.inverse else headCores
            module._SPARSE_LEFT, module._SPARSE_RIGHT = build_sparse_cores(allCores, len(entities))
            print("built {} compiled sparse cores per orientation".format(len(module._SPARSE_LEFT)),
                  flush=True)
        if args.pricing == "tnreason":
            headCores = K.relation_cores(rows, cols, rIdx, len(entities), len(relations))
            allCores = headCores + [c.T.tocsr() for c in headCores] if args.inverse else headCores
            import sys
            module = sys.modules[__name__]
            module._POLY_LEFT, module._POLY_RIGHT = build_poly_cores(allCores, len(entities))
            print("built {} polynomial cores per orientation for tnreason pricing".format(len(_POLY_LEFT)),
                  flush=True)
        cores, ruleSets, bodies, n, total = learn_rules(entities, relations, rows, cols, rIdx,
                                                        args.depth, args.rules, args.minSupport,
                                                        args.maxBodies, args.inverse, args.constants,
                                                        args.topology)
        if args.export:
            exportRules(ruleSets, relations, len(relations), args.export)
        trainIdx = list(zip(rows.tolist(), rIdx.tolist(), cols.tolist()))
        evaluate(cores, ruleSets, n, relations, entIndex, relIndex, trainIdx,
                 [os.path.join(args.dir, "valid.txt")], os.path.join(args.dir, "test.txt"),
                 args.maxTest, args.priorOnly, args.aggregation, "both", args.dumpRanks)
