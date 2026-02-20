from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Any

# -----------------------------
# Hilfsfunktionen: rXcY -> a_rb_r_cb_c_n
# -----------------------------

def parse_rc(rc: str) -> Tuple[int, int]:
    """Wandelt 'r{row}c{col}' (1-basiert) in (row, col) ints um."""
    if not (rc.startswith("r") and "c" in rc):
        raise ValueError(f"Ungültig: {rc}")
    r_str, c_str = rc[1:].split("c", 1)
    r = int(r_str)
    c = int(c_str)
    if not (1 <= r <= 9 and 1 <= c <= 9):
        raise ValueError(f"r/c ausserhalb 1..9: {rc}")
    return r, c

def rc_to_block_indices(r: int, c: int) -> Tuple[int, int, int, int]:
    """
    1-basierte (r,c) -> 0-basierte (rb, r_in, cb, c_in).
    rb, r_in, cb, c_in ∈ {0,1,2}
    """
    rb = (r - 1) // 3
    r_in = (r - 1) % 3
    cb = (c - 1) // 3
    c_in = (c - 1) % 3
    return rb, r_in, cb, c_in

def bool_var_name(rb: int, r_in: int, cb: int, c_in: int, n: int) -> str:
    """Erzeugt den Bool-Variablennamen a_{rb}_{r}_{cb}_{c}_{n} mit 0-basiertem n ∈ {0..8}."""
    if not (0 <= rb <= 2 and 0 <= r_in <= 2 and 0 <= cb <= 2 and 0 <= c_in <= 2):
        raise ValueError(f"Block-/Inblock-Index ausserhalb 0..2: {(rb, r_in, cb, c_in)}")
    if not (0 <= n <= 8):
        # Falls in den Slices jemand 9 benutzt hat, explizit fehlschlagen:
        raise ValueError(f"Digitindex n muss 0..8 sein (gesehen: {n})")
    return f"a_{rb}_{r_in}_{cb}_{c_in}_{n}"

def cell_handle_to_bool_family(cell_rc: str) -> List[str]:
    """
    Liefert alle 9 Bool-Variablen einer Zelle (n = 0..8).
    Nützlich, wenn man 'genau-eins' lokal modellieren will.
    """
    r, c = parse_rc(cell_rc)
    rb, r_in, cb, c_in = rc_to_block_indices(r, c)
    return [bool_var_name(rb, r_in, cb, c_in, n) for n in range(9)]

# -----------------------------
# Eingabe-Edge-Format
# -----------------------------

@dataclass
class Edge:
    name: str
    variables: Dict[str, str]        # z.B. {"A":"r5c9","B":"r4c8",...}
    shapes: int                      # 9 ** arity
    slice_iterator: List[Tuple[int, Dict[str, int]]]  # [(1, {"A":i, "B":j, ...}), ...]

# -----------------------------
# Übersetzung der Slices zu Bool-Assignments
# -----------------------------

def translate_edge_to_boolean_assignments(edge: Edge) -> List[List[str]]:
    """
    Für jeden erlaubten Slice (1, {Handle: idx, ...}) erzeuge die Liste der
    True-gesetzten Bool-Variablen ['a_rb_r_cb_c_idx', ...] — eine pro Handle.
    Rückgabe: Liste von Assignments; jedes Assignment ist eine Liste Bool-Variablen.
    """
    assignments: List[List[str]] = []

    # Vorverarbeitung: Handle -> (rb,r,cb,c)
    handle_to_block = {}
    for handle, rc in edge.variables.items():
        r, c = parse_rc(rc)
        handle_to_block[handle] = rc_to_block_indices(r, c)

    for value, mapping in edge.slice_iterator:
        if value != 1:
            # In unseren harten Constraints nutzen wir nur 1 (erlaubt)
            continue

        true_vars: Dict = {}
        vars: set = set()
        for handle, idx in mapping.items():
            if handle not in handle_to_block:
                raise KeyError(f"Handle {handle} nicht in edge.variables")
            rb, r_in, cb, c_in = handle_to_block[handle]
            vars_ = bool_var_name(rb, r_in, cb, c_in, idx)
            true_vars[vars_] = 1
            vars = vars.union(vars_)

        assignments.append((value,true_vars))
    final = {"slice_iterator":assignments, "variables": list(vars)}
    return final

def translate_edges(edges: Iterable[dict]) -> Dict[str, List[List[str]]]:
    """
    Nimmt eine Liste von Edge-Dicts (so wie aus deinem JSON) und liefert
    ein Dict: edge_name -> Liste erlaubter Bool-Assignments (je Assignment eine Liste True-Variablen).
    """
    out: Dict[str, List[List[str]]] = {}
    for e in edges:
        edge = Edge(
            name=e["name"],
            variables=e["variables"],
            shapes=e["shapes"],
            slice_iterator=[tuple(x) for x in e["slice_iterator"]],
        )
        # out[edge.name] = translate_edge_to_boolean_assignments(edge)
        sliceiter_vars = translate_edge_to_boolean_assignments(edge)
        out[edge.name] = {"name":edge.name,"variables":sliceiter_vars["variables"],"shapes":[2 for _ in range(len(sliceiter_vars["variables"]))],"slice_iterator":sliceiter_vars["slice_iterator"]}
    return out

# -----------------------------
# Beispiel
# -----------------------------
if __name__ == "__main__":
    # Beispiel: diagonale Summe 15 über 5 Zellen (dein Eintrag)
    example_edge = {
        "name": "diagonal_sum_2",
        "variables": {"A": "r5c9", "B": "r4c8", "C": "r3c7", "D": "r2c6", "E": "r1c5"},
        "shapes": 9**5,
        "slice_iterator": [
            (1, {"A": 0, "B": 0, "C": 0, "D": 2, "E": 8}),  # -> (1,1,1,3,9)
            (1, {"A": 4, "B": 0, "C": 0, "D": 2, "E": 4}),  # -> (5,1,1,1,5)
        ],
    }

    boolean_assignments = translate_edges([example_edge])
    # Ausgabe: pro erlaubtem Slice die Liste der True-Variablen a_rb_r_cb_c_n
    from pprint import pprint
    pprint(boolean_assignments["diagonal_sum_2"])
