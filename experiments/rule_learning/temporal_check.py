"""Does a time-ordering rule have support on a temporal event graph?

The numerical leg of the paper needs a real dataset where an ordering comparison between two events
carries signal. Before any modelling, this measures whether that is true on ICEWS14: enumerate two-hop
event chains, split by whether the second event is later than the first, and report how often a fact
exists between the endpoints, and with which time relation.

Usage:
    python temporal_check.py [--maxPairs 2000000]
"""

import argparse
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    facts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            facts.append((parts[0], parts[1], parts[2], parts[3]))
    return facts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=os.path.join(HERE, "data", "icews14"))
    parser.add_argument("--maxPairs", type=int, default=2000000)
    args = parser.parse_args()

    facts = load(os.path.join(args.dir, "train.txt"))
    entities = sorted({h for h, _, _, _ in facts} | {t for _, _, t, _ in facts})
    relations = sorted({r for _, r, _, _ in facts})
    dates = sorted({d for _, _, _, d in facts})
    eIdx = {e: i for i, e in enumerate(entities)}
    dateIdx = {d: i for i, d in enumerate(dates)}
    print("{} facts, {} entities, {} relations, {} distinct dates from {} to {}".format(
        len(facts), len(entities), len(relations), len(dates), dates[0], dates[-1]), flush=True)

    outOf = defaultdict(list)          # entity -> [(rel, tail, time)]
    inTo = defaultdict(list)           # entity -> [(rel, head, time)]
    factTime = defaultdict(set)        # (head, tail) -> times
    for h, r, t, d in facts:
        time = dateIdx[d]
        outOf[h].append((r, t, time))
        inTo[t].append((r, h, time))
        factTime[(h, t)].add(time)

    pairs = later = endpointExists = laterAgrees = 0
    sameWindow = 0
    forward = defaultdict(int)
    forwardHit = defaultdict(int)
    for z in entities:
        for r1, x, t1 in inTo[z]:
            for r2, y, t2 in outOf[z]:
                pairs += 1
                if pairs > args.maxPairs:
                    break
                hit = (x, y) in factTime
                if t1 < t2:
                    later += 1
                    if hit:
                        endpointExists += 1
                        if any(t3 > t2 for t3 in factTime[(x, y)]):
                            laterAgrees += 1
                elif t1 == t2:
                    sameWindow += 1
                if hit:
                    forward[(r1, r2)] += 1
                forwardHit[(r1, r2)] += 0
            if pairs > args.maxPairs:
                break
        if pairs > args.maxPairs:
            break

    density = len(facts) / float(len(entities) ** 2)
    capped = " (truncated at --maxPairs)" if pairs > args.maxPairs else ""
    print("two-hop event chains enumerated: {}{}".format(pairs, capped))
    print("of those, first event strictly before second: {} ({:.3f})".format(
        later, later / max(pairs, 1)), flush=True)
    print("endpoint fact exists: {} ({:.3f} of ordered chains, against entity-pair density {:.4f})".format(
        endpointExists, endpointExists / max(later, 1), density), flush=True)
    print("and the endpoint fact is later than both events: {} ({:.3f} of ordered chains with a fact)".format(
        laterAgrees, laterAgrees / max(endpointExists, 1)), flush=True)
    print("chains with simultaneous events: {}".format(sameWindow))

    ranked = sorted(((count, key) for key, count in forward.items()), reverse=True)[:10]
    print("\ntop relation pairs by endpoint facts:")
    for count, (r1, r2) in ranked:
        print("  {:40s} & {:40s} {}".format(r1, r2, count))


if __name__ == "__main__":
    main()
