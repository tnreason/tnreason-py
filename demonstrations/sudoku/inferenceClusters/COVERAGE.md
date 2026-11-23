# Rule coverage and tensor translations

This is a finite, researched catalogue, not a claim to cover every named Sudoku
method or every pattern within a family. The primary references below are
Bernhard Hobiger's published HoDoKu technique documentation and Andrew Stuart's
published explanations for his SudokuWiki solver. Links identify the source of
the rule definitions; tensor constructions and resource bounds are this
library's translations. Sources were checked on 2026-11-03. Ordinary Sudoku's
axioms require one of each digit in each row, column, and box; puzzle uniqueness
is a separate premise. See [Nikoli's published rules](https://www.nikoli.co.jp/en/puzzles/sudoku/).

## Reading the tables

Use `n = num**2`. Let `p(c)` be the position category of cell `c`, `u(H,d)` the
unit category choosing where digit `d` goes in house `H`, and `a(c,d)` its Boolean
indicator. `C` is a set of selected categories. Define:

* `A(C)` = all indicators occurring in at least two categories of `C`.
* `L(C,A)` = **every original tensor** with scope `{category, indicator}` for
  `category in C`, `indicator in A`, and the corresponding incidence in
  `get_sudoku_constraints(num)`. Multiple original tensors on a scope are all
  included. The default is `A=A(C)`.
* `E(C; O)` = `L(C,A(C))`, retaining output variables `O`. Every other variable
  in those original tensors is internal and existentially eliminated.

The tables specify the exact categories, indicators where overridden, and
outputs used by implemented selectors. `.core_keys` identifies the actual
original tensors; `.participating_variables` and `.internal_variables` make
their scopes inspectable. In all implementations, indicators and categories
receive their current domain masks, and conclusions come only from exact
Boolean-support projection. Detectors recognize geometry and domains; they do
not directly remove a textbook candidate.

**B** denotes the common hard execution bounds: selected original tensors
`<= max_cores`, cumulative dense work `<= max_work`, each input/intermediate
`<= max_size`, each expanded output `<= max_output_size`, and planning
factor-scope examinations `<= max_planning_steps`. Exceeding any
bound returns `skipped_budget`, not an elimination. The default limits are
256 tensors, 10,000,000 work units, 1,000,000 intermediate entries, and 100,000
entries per output; the separate planning budget defaults to 10,000,000 scope
examinations. Fixed axes are compacted; local degree-one/two Boolean atoms are eliminated
early. A verified greedy bucket path handles the remaining variables. See [README.md](README.md)
for all detection, cache, board, network, and rank bounds.

For a standard network with one link per incidence, `L(C,A)` has exactly
`sum(category in C, |A intersect atoms(category)|)` original tensors, each of
size `2n`. In particular it has at most `n * |C|` tensors. A unary category
output has `n` expanded entries; a joint output of `q` categories has `n**q`.
These are output sizes, not a claim of polynomial total contraction work.
Actual intermediate scopes determine whether execution fits **B**.

“Implemented” means there is a bounded selector and exact contraction interface
for the specified portion. “GAC-equivalent” is specific to this repository's
binary category/indicator model. “Assumption-gated” means an additional axiom
is absent and execution is rejected; it does **not** mean an enabled rule is
available behind a Boolean flag. “Unsupported” means that only a possible
translation and the remaining work are described.

## Implemented selectors

| Family, equivalent names, and source | Exact recognized preconditions and intended elimination | Exact selected cluster and outputs | Bounds and completed scope |
| --- | --- | --- | --- |
| Naked single; hidden single; full house/last digit. [HoDoKu singles](https://hodoku.sourceforge.net/en/tech_singles.php) | A position category or unit-digit category has exactly one current option. Force its chosen indicator true and the others false. A full house is a single arising after the other cells in a house are fixed. | `C={the fixed category}`, explicitly `A=all n of its indicators`, `O=A`. The category is internal; all original incident links are retained. | **GAC-equivalent**, implemented `detect_singles`. `n` original links and `n` Boolean unary outputs. `max_steps` bounds visited categories; **B** bounds execution. |
| Locked candidates type 1/pointing and type 2/claiming; line-box interaction. [HoDoKu intersections](https://hodoku.sourceforge.net/en/tech_intersections.php) | For digit `d`, all source choices lie in a line/box intersection, at least two remain, and the target has an outside choice. Pointing uses box as source; claiming uses line. Exclude target placements outside the intersection. | `C={u(line,d),u(box,d)}`, explicitly `A=their complete intersection`, `O={target unit category}`. | **Implemented** `detect_locked`; rows and columns, both directions. Exactly `2*num` links in the standard encoding. Singleton sources belong to singles. **B**. |
| Naked subsets: pair, triple, quad/quadruple; locked pair/triple when also contained in a second house. [HoDoKu naked subsets](https://hodoku.sourceforge.net/en/tech_naked.php) | In one house, `k` cells each have 2..`k` options whose union is exactly `k` digits. Exclude these digits from the other cells in that house. Candidate sets need not be identical. | `C={p(c): c in chosen cells} union {u(H,d): d in the union}`, `O=C`. `A(C)` is the complete `k` by `k` incidence rectangle, including currently absent choices. | **Implemented** `detect_subsets(hidden=False)`, configurable `2 <= k <= max_subset_size`, `k<n`; exactly `2*k*k` links. Scans all three house types; a second-house locked subset is a separate eligible selection. Singleton-containing degeneracies are omitted. **B**. |
| Hidden subsets: pair, triple, quad/quadruple. [HoDoKu hidden subsets](https://hodoku.sourceforge.net/en/tech_hidden.php) | In one house, `k` digits each have 2..`k` positions whose union is exactly `k` cells. Exclude the other digits from those cells. | Same complete incidence rectangle as naked subsets; `C` contains the `k` unit-digit and `k` position categories, `O=C`. | **Implemented** `detect_subsets(hidden=True)` with the same parameter bounds, house symmetries, `2*k*k` links, and omitted singleton degeneracies. **B**. |
| Basic fish: X-Wing (`k=2`), Swordfish (3), Jellyfish (4); larger parameterized fish. [HoDoKu basic fish](https://hodoku.sourceforge.net/en/tech_fishb.php) | For one digit, `k` base rows each have 2..`k` choices and their union is exactly `k` cover columns, or transposed. Remove that digit in covers outside bases. Missing body corners are allowed. | `C={k base unit-digit categories} union {k cover unit-digit categories}`, `O=cover categories`; `A(C)` is the `k*k` base/cover grid. | **Implemented** `detect_fish(kind="basic")`, `2 <= k <= max_fish_size`, `k<n`; `2*k*k` links. Default max 4, so larger names require explicitly raised limits. Degenerate bases containing a single are omitted. **B**. |
| Finned basic fish: finned X-Wing/Swordfish/Jellyfish. [HoDoKu finned fish](https://hodoku.sourceforge.net/en/tech_fishfs.php) | `k` row/column bases and covers; all extra base choices (fins) lie in one box. Every base has at least two body choices. Targets lie in covers outside bases and in the fin box, so see every fin. | `E(C; O)` where `C` is the `2k` base/cover categories plus `u(fin box,d)`; `O` is each cover category with an eligible target in that box. All shared atoms, including fins and box/cover intersections, enter. | **Implemented restricted selector** `detect_fish(kind="finned")`; both orientations, `k <= max_fish_size`, **B**. Does not cover multi-box fins, endo-fins, mixed-house fish, or base singletons. |
| Sashimi fish: sashimi X-Wing/Swordfish/Jellyfish. [HoDoKu sashimi fish](https://hodoku.sourceforge.net/en/tech_fishfs.php) | Same fin conditions, but at least one base has exactly one body choice; every base has at least one body choice and at least two total choices. Same targets/elimination. | Identical finned construction and outputs; a different geometry predicate names the family. | **Implemented restricted selector** `detect_fish(kind="sashimi")`, same bounds. Broader source definitions also call other degenerate fin-false bodies sashimi; those without a singleton base are not separately recognized by this name here. **B**. |
| XY-Wing/Y-Wing. [HoDoKu wings](https://hodoku.sourceforge.net/en/tech_wings.php) | Pivot `{X,Y}`, two peer pincers `{X,Z}` and `{Y,Z}`, distinct digits. Remove `Z` from a different cell seeing both pincers. | `E(C; {p(target)})`; `C` contains all four position categories and every `u(H,d)` connecting pivot/pincer on `X` or `Y` and each pincer/target on `Z`. | **Implemented** `detect_xy_wings`; all ordinary peer geometries, including transposes. Detection bounded by `max_steps`, `max_patterns`; **B**. |
| XYZ-Wing. [HoDoKu wings](https://hodoku.sourceforge.net/en/tech_wings.php) | Pivot `{X,Y,Z}` and pincers `{X,Z}`, `{Y,Z}` seeing pivot. Target must see both pincers **and pivot**; remove `Z`. | XY-Wing construction, additionally including each `u(H,Z)` connecting pivot and target. Output is the target position. | **Implemented** `detect_xyz_wings`; all ordinary peer geometries and the same detector/execution bounds. |
| Simple coloring: color trap/color wrap; single-digit conjugate paths. [HoDoKu coloring](https://hodoku.sourceforge.net/en/tech_col.php) | Each path edge is a unit with exactly two current positions for one digit. A target outside the path seeing opposite-color endpoints loses the digit (trap). Same-color endpoints seeing each other exclude that path parity (wrap). | `E(C; O)`; `C` contains path positions, the target for a trap, conjugate-edge unit categories, and all shared unit categories proving the endpoint/target conflicts. Trap `O={p(target)}`; wrap `O={p(c): c at even path indices}`. | **Implemented bounded path portion**, `detect_simple_coloring`; length in candidate cells `<= max_chain_length`, scan/pattern caps and **B**. This does not enumerate complete arbitrary coloring components or every wrap consequence. |
| XY-Chain; remote pairs as same-bivalue special cases; XY-Wing as a short instance. [HoDoKu chains](https://hodoku.sourceforge.net/en/tech_chains.php) | A simple path of bivalue cells uses the outgoing digit of one cell as the incoming digit of its peer. Endpoints have the same unlinked digit `Z`. Target outside path sees both endpoints; remove `Z`. | `E(C; {p(target)})`; `C` contains path and target positions, plus all shared unit categories for each linking digit and both endpoint/target `Z` conflicts. | **Implemented** `detect_xy_chains`; 3..`max_chain_length` cells, repeated cells excluded. Remote-pair elimination is represented by eligible even-length paths. No separate named selector is claimed. Scan caps and **B**. |
| ALS-XZ, ordinary singly linked elimination. [HoDoKu ALS](https://hodoku.sourceforge.net/en/tech_als.php) | Disjoint sets `A,B` each occupy one house; each set has its own `k` unsolved cells and `k+1` digits. All cross-set occurrences of shared digit `X` see each other (RCC). Target outside both sets sees every occurrence of another common digit `Z`; remove `Z`. | `E(C; {p(target)})`; `C` contains positions in `A,B,target`, each ALS house category for each digit in its union, and all shared unit categories proving cross-set `X` and source/target `Z` conflicts. | **Implemented restricted selector** `detect_als_xz`; 1..`max_als_size` cells per ALS, scan caps and **B**. This legacy selector omits overlap and special doubly-linked deductions; `detect_als_xz_extended` implements them below. |

The local selectors in the preceding table require a still-present target,
except singles, which describe already-fixed categories. Pattern sizes or path lengths
outside configured bounds are outside coverage even if `truncated=False`.
An unpropagated context can miss patterns whose position and unit views would
become consistent after GAC; no propagation is hidden in detection.

## Relationship to ordinary GAC

The original model consists of binary category/indicator links. GAC on those
links already propagates singles, including full houses after the relevant
restrictions have reached their categories. The single selector is useful for
descriptions and exact support tests, but adds no propagation power at GAC
closure.

Joining several links can add power: locked candidates, subsets, fish, wings,
bounded coloring/chains, and the supported ALS-XZ patterns can exclude values
left by GAC on the individual original links. This does not mean each detected
instance must improve an already-closed network: an easier deduction, an extra
constraint, or another cluster may already subsume it. The tests include cases
that establish the additional power. If Sudoku is instead modeled with a
single global `alldifferent` constraint per house, exact GAC on that global
constraint already includes naked and hidden Hall subsets. Comparing against
that stronger model must not be described as comparing against the binary
GAC implemented here.

## Expanded ordinary-rule coverage

The following selectors are also implemented. Their configured bounds and
precise portions matter: “implemented” does not mean exhaustive discovery of
all sizes, group geometries, or proof networks. Named wrappers restrict shared
mechanisms instead of manufacturing a different tensor relation for each name.

Additional exact construction notation:

* `I(Q; O)` selects **all n original incidence links** of every category in
  `Q`, including currently absent options, and retains `O`. It has `n*|Q|`
  binary links in the standard encoding. Used for implication premises.
* `M(S; O)` selects the current candidate indicators in cells `S`, each
  indicator's position and all three ordinary house categories, and all four
  original incident links. It retains the requested position categories `O`.
  With one original link per incidence it has exactly
  `4 * sum(c in S, |options(p(c))|)` links, bounded by `4*n*|S|`.
  Currently absent options are omitted from indicator selection. Domain masks
  enter only for participating variables, so an absent indicator whose mask
  has not propagated into its category can make this local projection weaker.
  A GAC-closed input exposes more deductions; omitted links never justify an
  unsupported elimination.
* `U(clusters; O)` is the deduplicated union of their original tensor keys,
  restricted afresh by the executing context's domains. It never contracts
  previously derived summaries as if they were original constraints.

Every construction obeys **B**. A recognized geometry may yield no extra
restriction under exact support, especially when the supplied tensor supports
are changed. The detector's logical reasoning only selects a proof; it never
supplies an elimination mask in place of contraction.

| Family and published definition | Exact selection/preconditions and intended consequence | Original-tensor construction, scope and cost bounds |
| --- | --- | --- |
| X-Chains, Turbot Fish, Skyscraper, 2-String Kite. [HoDoKu single-digit patterns](https://hodoku.sourceforge.net/en/tech_sdp.php), [chains](https://hodoku.sourceforge.net/en/tech_chains.php) | Same-digit alternating weak/strong paths. Turbot uses two disjoint conjugate pairs and a weak bridge. Skyscraper restricts conjugates to parallel lines and bridge to a perpendicular line; kite uses row/column conjugates with a box bridge. A target outside the four cells seeing both outer endpoints loses the digit. | **Implemented** `detect_x_chains`, `detect_turbot_fish`, `detect_skyscrapers`, `detect_two_string_kites`. `I(Q; {target atom})` for conjugates, bridge and endpoint conflicts. General paths have at most `max_chain_length` edges; all graph/scan caps apply. |
| Empty Rectangle, including two-candidate and grouped arms. [HoDoKu single-digit patterns](https://hodoku.sourceforge.net/en/tech_sdp.php) | A box's current digit choices lie on a row/column cross with absent intersection and both arms nonempty. An external line conjugate has its near endpoint on one cross arm's line; its far endpoint conflicts with the target on the other arm's line. Transposition included. | **Implemented** `detect_empty_rectangles`. `I(Q; {target atom})` uses box, external conjugate, arm bridge and target-conflict categories. The two dual instances are independently selectable. Scan/pattern caps and **B**; no invented grouped axiom. |
| W-Wing. [HoDoKu wings](https://hodoku.sourceforge.net/en/tech_wings.php) | Two identical bivalue cells `{X,Y}` see opposite ends of an external conjugate on `X`. A target outside the four cells sees both bivalue cells; eliminate `Y`. | **Implemented** `detect_w_wings`. `I(Q; {target atom})`: the two positions, conjugate unit, connecting `X` categories and target `Y` conflicts. All pair/bridge/target trials charged to the scan. |
| Generalized bent subsets; WXYZ-Wing. [SudokuWiki WXYZ-Wing](https://www.sudokuwiki.org/WXYZ_wing) | `k` unsolved cells have exactly `k` distinct digits. Every digit except one is restricted: all its occurrences see each other. Target sees every occurrence of the sole nonrestricted digit. WXYZ uses four cells/four digits coverable by two ordinary houses. | **Implemented** `detect_bent_subsets`, `detect_wxyz_wings`, validated `describe_bent_subset`. `M(S plus target; {p(target)})`. `3 <= k <= max_subset_size`; WXYZ specifically 4. Covers non-hinge geometries meeting this predicate; other uses of the same wing names are not automatically claimed. |
| AICs, X-Cycles, continuous/discontinuous nice loops. [HoDoKu chains](https://hodoku.sourceforge.net/en/tech_chains.php) | Strong links join the only two remaining choices of a cell/unit; weak links join mutually exclusive choices. Alternating literal reachability from a candidate to its complement refutes that starting truth value. A continuous even alternating cycle supplies weak-link common-conflict exclusions. | **Implemented** `detect_aic`, `detect_nice_loops`; registry `x_cycles` restricts to one digit. `I(Q; O)` for every path/cycle premise and target conflict. Outputs are original Boolean indicators, so both placements and exclusions are supported. Each path/cycle has `<= max_chain_length` edges; the whole graph has separate node/edge caps. Scan/pattern caps and **B** also apply. |
| Grouped AICs/grouped nice loops. [HoDoKu grouped links](https://hodoku.sourceforge.net/en/tech_chains.php) | A node is an OR of at most `max_group_size` candidates. Two disjoint groups form a strong edge when they partition all current choices of a category. A weak edge requires their union to belong to one common exactly-one category. | **Implemented** `detect_grouped_aic` and `detect_nice_loops(grouped=True)`. Same `I(Q; O)` construction, retaining original singleton target atoms. No auxiliary assumed group tensor is introduced. Mixed-house Cartesian/group exclusions lacking a single common category and ALS-as-graph-node edges remain outside this graph predicate; ALS chains have a separate implementation. |
| Multi-colors types 1/2. [HoDoKu coloring](https://hodoku.sourceforge.net/en/tech_col.php) | Two distinct single-digit conjugate components have a weak conflict between colors. Opposite colors yield an at-least-one alternative for a target trap; a color conflicting with both colors of the other component is false. | **Implemented** `detect_multi_coloring`. `I(Q; O)` combines both conjugate trees, bridge/conflict premises and targets. Two components per selection, each at most `max_coloring_nodes`; graph/scan caps and **B**. Arbitrary simultaneous multi-component nets use the forcing-proof interface. |
| 3D Medusa. [SudokuWiki Medusa](https://www.sudokuwiki.org/3D_Medusa) | Unit conjugates and bivalue-position strong edges define multidigit two-color components. Same-color mutual conflicts or a color emptying a cell/unit refute that color; a candidate weakly linked to both colors is false. Odd conjugate cycles are contradictory. | **Implemented** `detect_medusa`. `I(Q; O)` includes the conjugate tree and every conflict/exhausted-category premise. Outputs are refuted-color or trapped original indicators. `max_coloring_nodes`, graph/scan caps and **B**. Coloring trees use the component-size bound, rather than the implication path-length bound. No uniqueness premise. |
| Overlapping ALS-XZ and doubly linked ALS. [HoDoKu ALS](https://hodoku.sourceforge.net/en/tech_als.php) | Every ALS has `k` unsolved cells in a house and `k+1` digits. An RCC's cross-set occurrences all see each other and it must be absent from overlap. Singly linked XZ target sees every common non-RCC digit occurrence. With two distinct RCCs, RCCs lock into the union and each non-RCC locks into its individual ALS; valid cannibalistic targets in the other ALS are included. | **Implemented** `detect_als_xz_extended(kind="singly"/"doubly"/"both")`, `describe_als_xz`, `describe_doubly_linked_als`. `M(A union B plus target; {p(target)})`. Per-set `max_als_size`, catalogue `max_als_sets`, scan/pattern caps and **B**. |
| ALS-XY-Wing and ALS chains. [HoDoKu ALS](https://hodoku.sourceforge.net/en/tech_als.php) | Distinct ALS nodes have valid RCC edges; consecutive RCC digits differ. Endpoint elimination digit is common to endpoints, differs from each endpoint's incident RCC, and every occurrence at both endpoints sees the external target. Overlap away from edge RCCs is permitted. | **Implemented** `detect_als_chains`, `detect_als_xy_wing`, `describe_als_chain`. `M(union of ALS cells plus target; {p(target)})`. Wing has 3 ALS nodes; selected lengths run through `max_chain_length`, bounded by the finite catalogue. All selection and traversal is charged. **B**. |
| Death Blossom. [HoDoKu ALS](https://hodoku.sourceforge.net/en/tech_als.php) | One ALS petal per current stem digit; stem sees all occurrences of that digit in its petal, and is outside every petal. A digit common to all petals is excluded from an external target seeing all petal occurrences; if that digit is also in the stem, target must see the stem. | **Implemented** `detect_death_blossom`, `describe_death_blossom`. `M(stem union petals plus target; {p(target)})`. Exactly 2..`max_petals` exhaustive stem alternatives; petals `<= max_als_size`, catalogue cap, scan/pattern caps and **B**. Overlapping petals are allowed subject to the stated premises. |
| Franken/Mutant fish, multi-box fins, endo-fins, cannibalistic fish. [HoDoKu general fish](https://hodoku.sourceforge.net/en/tech_fishg.php), [complex fish](https://hodoku.sourceforge.net/en/tech_fishc.php) | Equal-sized base/cover sets contain distinct houses, with no house in both sets. Franken mixes a line orientation with boxes; Mutant permits arbitrary house types. Uncovered base choices are exo-fins; base multiplicity above one gives endo-fins. An ordinary target is covered and outside the current base-candidate set; a cannibalistic target is a base candidate in multiple covers. Target must see every fin. | **Implemented** `detect_general_fish` and taxonomy/fin wrappers. `E(C; {p(target)})` for base/cover categories, target position and every original target/fin conflict category. Candidate multiplicities are inspected explicitly. Fin-false degeneracies are allowed; contraction decides the result. Fin count `<= max_fins <= n*n`, house count `<= min(n,max_fish_size)`, scan/pattern caps and **B**. Explicit house selections avoid prefix starvation. |
| Siamese fish. [HoDoKu complex fish](https://hodoku.sourceforge.net/en/tech_fishc.php) | Two same-taxonomy finned components have identical base sets and cover sets differing by exactly one house, with the same occupied base candidate cells. Each component has a valid fin-seeing elimination; combine distinct target outputs. | **Implemented** `detect_siamese_fish`, optional explicit `cover_pairs`. `E(C1 union C2; both target positions)`, including any additional original links shared across that category union. Intermediate components `<= max_components`, default `min(2*max_patterns,max_steps)`; common pair/scan caps and **B**. |
| Sue de Coq/two-sector disjoint subsets. [HoDoKu miscellaneous](https://hodoku.sourceforge.net/en/tech_misc.php) | Disjoint unsolved intersection `I`, box sector `B`, line sector `R`; digit sets `V,VB,VR`; `V intersect VB intersect VR` is empty, side excesses `eB=|VB|-|B|` and `eR=|VR|-|R|` positive, and `|I|=eB+eR+|V minus VB minus VR|`, with `|V|>=|I|+2`. Exclude `VB union (V minus VR)` from the remaining box and `VR union (V minus VB)` from the remaining line. | **Implemented** basic and generalized counting `detect_sue_de_coq(extended=...)`, `describe_sue_de_coq`. `M(I union B union R plus target; {p(target)})`. Targets exclude `I` and the corresponding side's cells. Intersection `<= min(num,max_subset_size)`; each side `<= max_als_size`, side catalogue `<= max_als_sets`. Outside-`V` digits may occur in both sectors. Scan/pattern caps and **B**. |
| Aligned pair/triple exclusion; generalized aligned exclusion. [SudokuWiki APE](https://www.sudokuwiki.org/Aligned_Pair_Exclusion) | `q` unsolved base cells and house-contained helper sets of `k` cells with `k+1`..`k+q-1` digits. Every base sees every helper cell; each helper's digits intersect the union of base candidates in at least `excess+1` digits. Original conflicts reject base tuples starving a helper; projection excludes base values absent from all supported tuples. | **Implemented** `detect_aligned_exclusion(size=2/3/...)`, `describe_aligned_exclusion`. `M(base union helpers; base positions)`. Bases `<= max_subset_size`; helper cells `<= max_als_size`, at most `max_petals` helper sets; catalogue/scan caps and **B**. Helpers are disjoint from the base but may overlap each other. Bases need not be mutual peers. No hand-coded tuple rejection. |
| Templates/single-digit pattern support. [HoDoKu templates](https://hodoku.sourceforge.net/en/tech_last.php) | For one digit retain placements satisfying all its row, column and box exactly-one constraints under current domains. An unsupported placement is false; a unique surviving row placement is forced. | **Implemented** `detect_templates`, `describe_templates`. All `3n` unit categories and all `n*n` digit indicators, with `3n*n` original links; default outputs the `n` row categories. No template enumeration. Scan/board caps and **B**; a broad 9x9 projection may be skipped by execution budgets. |
| Multi-digit pattern overlay. [HoDoKu template principle](https://hodoku.sourceforge.net/en/tech_last.php) | Overlay at least two complete digit-placement relations and enforce cell exclusivity between those digits. Exclude placements absent from every compatible overlay. | **Implemented** `detect_pattern_overlay`, `describe_pattern_overlay`. Add every position category to selected digits' template categories; explicitly select all indicators for those digits, retaining requested row categories. Exactly `4*len(digits)*n*n` original incidence links. Smaller requested `outputs` reduce work; all projections obey **B**. |
| Forcing chains and finite forcing nets. [HoDoKu last resort](https://hodoku.sourceforge.net/en/tech_last.php) | Trace each current source alternative through original exactly-one premises. Common conclusions across all viable alternatives follow; a refuted alternative requires an actual contradiction proof. Nets can combine several false premises to leave a category's final candidate; chains restrict such steps to original conjugate pairs. | **Implemented bounded proof selection** in `forcing.py`; `I(Q; {consequence atom})` includes source and every proof category, and contraction rederives the consequence. Automatic sources have 2..`max_petals` alternatives, depth `<= max_chain_length`, literal/premise/scan/pattern caps and **B**. Explicit `describe_forcing_net` checks exhaustive source-key coverage and uses `U` plus complete source incidences; it does not assume caller-stated implications. No nested candidate search or solver calls. |
| Kraken fish types 1/2. [HoDoKu Kraken fish](https://hodoku.sourceforge.net/en/tech_last.php) | Type 1: every exo/endo-fin implies that a covered, non-base fish-digit target is false; the fin-false case excludes it directly. Type 2: every fin and every base candidate in a selected cover imply a common target false. The target may have another digit. | **Implemented compound original proofs** in `forcing.py`: `U(fish original region plus all implication proofs; target atoms)`. Descriptors check exhaustive premise coverage and original provenance; exact contraction establishes the consequence. Automatic queries require all relevant `premise=true -> target=false` paths within the shared graph/path/scan limits. Fish/fin/proof/target limits and **B** apply; detailed limits follow below. |
| Nishio-style single-digit contradiction. [HoDoKu templates/forcing](https://hodoku.sourceforge.net/en/tech_last.php) | A placement with no completion of its single-digit row/column/box relation is refuted. | **Implemented** `detect_nishio` uses the same exact original relation as templates, without trial-assignment search or a completed-grid lookup. It is a support-equivalent route to single-digit consistency, not a separate assertion of all historical Nishio procedures. Same template bounds and **B**. |

Automatic forcing traces cap the total active-indicator membership and each
branch's derived literals by `max_graph_nodes`; original-category premises are
bounded by `max_net_categories`. An incomplete branch makes its source unknown
and marks detection truncated. An empty source can produce a contradiction
cluster. Explicit forcing-net descriptions instead cap source alternatives by
`max_petals` and the total branch clusters plus the source cluster by
`max_patterns`; arbitrary supplied proof unions have no inferred trace depth.
They remain subject to exact execution bounds.

Automatic Kraken type 1 requires at least one fin and excludes cannibalistic
targets. Type 2 permits the fin-free case and considers one cover at a time.
Both retain currently undecided target indicators with domain `(0,1)` and
exclude a target that is itself a case premise. Fins are bounded by
`max_fins <= n*n`, house count by `min(n,max_fish_size)`, optional explicit
targets by `max_graph_nodes`, and case-proof count plus the fish body by
`max_patterns`. `grouped=True` uses the same bounded group partitions as the
implication graph. A missing path leaves the target unrecognized; graph or
scan exhaustion marks detection truncated. These selectors do not enumerate
arbitrary nested forcing networks.

## General proof interfaces and scope

`detect_proof_regions` proposes radius-limited connected original regions;
it does not claim a human rule was recognized. `describe_complete_support` /
`detect_complete_support` select the entire original network. Every logical
consequence of the finite encoded constraints is, in principle, an exact
projection of that relation. This gives a general translation route for any
finite original proof, including techniques not named in this catalogue.
It does not promise tractable full-network inference: **B** still applies and
recognition can be followed by an explicit execution skip. No search solver is
used as a fallback. Named selectors, proof descriptions and exact execution
remain separate interfaces.

All automatic catalogue/graph/path traversals have finite step and storage
limits. When a catalogue or scan is truncated, unvisited patterns remain
unclassified. A complete scan under a size/length bound does not establish
coverage outside that bound. Grouped weak links requiring a nonlocal Cartesian
exclusion, unrestricted ALS/AIC hybrids, arbitrary branch nesting, and larger
compound proof shapes are not automatically enumerated; explicit original
proof unions or complete support can express them within execution budgets.

## Additional axioms required

The following rules cannot safely be enabled from ordinary row/column/box
constraints alone. A bare assumption flag is rejected. Supplying a genuine
additional original constraint allows `describe_tensors` / complete support to
project it; that does not certify the external premise or implement a named
variant detector.

| Family and source | Missing premise and current behavior |
| --- | --- |
| Unique/hidden rectangles, BUG/BUG+1. [HoDoKu uniqueness](https://hodoku.sourceforge.net/en/tech_ur.php) | Require justified uniqueness. Ordinary tensors may support both swappable assignments, so their support alone cannot justify discarding them. Registry names remain explicitly assumption-gated; the library neither fabricates a uniqueness tensor nor looks up a completed solution. |
| Avoidable rectangles. [HoDoKu uniqueness](https://hodoku.sourceforge.net/en/tech_ur.php) | Require uniqueness and given/non-given provenance. Domains alone cannot distinguish a clue from a derived placement. Explicitly gated. |
| Diagonal/X-Sudoku, killer cages, anti-knight and other variants. [Sudoku X](https://www.sudokuwiki.org/SudokuX.aspx), [Killer Sudoku](https://www.sudokuwiki.org/Daily_Killer_Sudoku.aspx) | Require their original extra house/arithmetic/conflict tensors and metadata. Complete-support/original-proof projection includes supplied tensors, but no automatic variant-specific detector or implicit variant peer relation is installed. Cage sums use printed digits `internal+1`. |

For a particular puzzle already uniquely constrained by its original givens,
complete-support projection can derive the same placements that a uniqueness
technique would reach. The gate concerns the extra premise used by the local
uniqueness argument on an arbitrary supplied network, rather than an inability
to represent the consequences of an actually unique original network.

## Validation map

The new focused tests are in `tests/`; existing generic MAC, pair-cluster,
locked-cluster, and candidate-evaluation tests remain independent. Implemented
families are checked with expected removals and invalid near-patterns, exact
Boolean support checks, and GAC comparisons where the family adds power.
Cross-cutting tests cover transposes, signed and realized sparse support,
contradictions, immutable inputs, different domain snapshots, scan/execution
budgets, and `push_level`/`pop_level` application. These are focused fixtures,
not proof of exhaustive pattern discovery or a full Sudoku benchmark result.
