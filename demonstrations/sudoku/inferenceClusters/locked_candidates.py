def detect_locked_candidates_row(propagator, num=3):
    """
    Detect locked candidates (row -> square):
    if in a row, a number can only be in one square, then that number is locked to that square.
    """
    locked_candidates = []
    row_constraint_inds = [(c1, c2) for c1 in range(num) for c2 in range(num)]

    for r1 in range(num):
        for r2 in range(num):
            for n in range(num ** 2):
                row_key = f"row_{n}_{r1}_{r2}"
                possible_pos_flat = [i for i in range(num ** 2) if propagator.meanParamDict[row_key][{row_key: i}] > 0.5]
                possible_positions = [row_constraint_inds[ind] for ind in possible_pos_flat]

                if not possible_positions:
                    continue

                square_cols = set(pos[0] for pos in possible_positions)
                inner_cols = set(pos[1] for pos in possible_positions)

                if len(square_cols) == 1 and len(inner_cols) > 1:
                    locked_c1 = list(square_cols)[0]
                    square_key = f"square_{n}_{r1}_{locked_c1}"
                    locked_candidates.append([row_key, square_key] + [
                            "a_" + str(r1) + "_" + str(r2) + "_" + str(locked_c1) + "_" + str(c2) + "_" + str(n) for c2 in list(inner_cols)
                            ])
                    # print(f"Locked candidates detected: {row_key} number {n} locked in square {r1}_{locked_c1}")

    return locked_candidates

def detect_locked_candidates_col(propagator, num=3):
    """
    Detect locked candidates (col -> square):
    if in a column, a number can only be in one square, then that number is locked to that square.
    """
    locked_candidates = []
    col_constraint_inds = [(r1, r2) for r1 in range(num) for r2 in range(num)]

    for c1 in range(num):
        for c2 in range(num):
            for n in range(num ** 2):
                col_key = f"col_{n}_{c1}_{c2}"
                possible_pos_flat = [i for i in range(num ** 2) if propagator.meanParamDict[col_key][{col_key: i}] > 0.5]
                possible_positions = [col_constraint_inds[ind] for ind in possible_pos_flat]

                if not possible_positions:
                    continue

                square_rows = set(pos[0] for pos in possible_positions)
                inner_rows = set(pos[1] for pos in possible_positions)

                if len(square_rows) == 1 and len(inner_rows) > 1:
                    locked_r1 = list(square_rows)[0]
                    square_key = f"square_{n}_{locked_r1}_{c1}"
                    locked_candidates.append([col_key, square_key] + [
                            "a_" + str(locked_r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) for r2 in list(inner_rows)
                            ])
                    # print(f"Locked candidates detected: {col_key} number {n} locked in square {locked_r1}_{c1}")

    return locked_candidates

def detect_locked_candidates_square(propagator, num=3):
    """
    Detect hidden singles: if in a square, a number can only be in one row or one column.
    Returns list of tuples: (square_key, number, 'row'/'col', row/col_index)
    """
    locked_candidates = []
    # square_constraint = ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) for r2 in range(num) for c2 in range(num)]
    square_constraint_inds = [(r2,c2) for r2 in range(num) for c2 in range(num)]

    for r1 in range(num):
        for c1 in range(num):
            for n in range(num**2):
                square_key = f"square_{n}_{r1}_{c1}"
                possible_pos_flat = [i for i in range(num**2) if propagator.meanParamDict[square_key][{square_key: i}] > 0.5]
                possible_positions = [square_constraint_inds[ind] for ind in possible_pos_flat]
                
                if not possible_positions:
                    continue
                
                # Check rows
                rows = set(pos[0] for pos in possible_positions)
                cols = set(pos[1] for pos in possible_positions)
                if len(rows) == 1 and len(cols) > 1:
                    locked_row = list(rows)[0]
                    locked_candidates.append([square_key,
                                            "row_" + str(n) + "_" + str(r1) + "_" + str(locked_row),
                                            ]+[
                                            "a_" + str(r1) + "_" + str(locked_row) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
                                                for c2 in list(cols)
                                            ])
                    # print(f"Locked candidates detected: {square_key} number {n} locked in row {r1}_{locked_row}")
                
                # Check columns
                if len(cols) == 1 and len(rows) > 1:
                    locked_col = list(cols)[0]
                    locked_candidates.append([square_key,"col_" + str(n) + "_" + str(c1) + "_" + str(locked_col),]+[
                                            "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(locked_col) + "_" + str(n)
                                                for r2 in list(rows)
                                            ])
                    # print(f"Locked candidates detected: {square_key} number {n} locked in col {c1}_{locked_col}")
    
    return locked_candidates