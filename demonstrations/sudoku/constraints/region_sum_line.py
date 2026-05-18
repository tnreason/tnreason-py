from demonstrations.sudoku.constraints._constraint_utils import digit_count, digit_predicate_core, sum_to_hidden_core


def region_sum_segment_constraint(
    position_vars,
    sum_var,
    min_sum,
    max_sum,
    sudokuNum=3,
    coreType=None,
    max_dense_cells=5_000_000,
):
    return sum_to_hidden_core(
        sum_var=sum_var,
        position_vars=position_vars,
        min_sum=min_sum,
        max_sum=max_sum,
        sudokuNum=sudokuNum,
        name=f"region_sum_{sum_var}_{'_'.join(position_vars)}",
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )


def region_sum_line_constraints(
    segments,
    sum_var="region_sum",
    sudokuNum=3,
    min_sum=0,
    max_sum=None,
    prefix="region_sum",
    coreType=None,
):
    """
    Encode a region-sum line by grouping its cells into region segments.

    ``segments`` is a list of position-var lists, one list per box/region
    touched by the line. All segment sums are tied to the same hidden
    ``sum_var``.
    """
    segments = [list(segment) for segment in segments if segment]
    if max_sum is None:
        max_sum = max((len(segment) for segment in segments), default=0) * digit_count(sudokuNum)
    return {
        f"{prefix}_{index}": region_sum_segment_constraint(
            segment,
            sum_var,
            min_sum=min_sum,
            max_sum=max_sum,
            sudokuNum=sudokuNum,
            coreType=coreType,
        )
        for index, segment in enumerate(segments)
    }


def same_region_sum_constraint(segments, sudokuNum=3, coreType=None, max_dense_cells=5_000_000):
    segments = [list(segment) for segment in segments if segment]
    position_vars = [posVar for segment in segments for posVar in segment]
    segment_slices = []
    offset = 0
    for segment in segments:
        segment_slices.append(slice(offset, offset + len(segment)))
        offset += len(segment)

    def is_valid(*values):
        sums = []
        for segment_slice in segment_slices:
            sums.append(sum(value + 1 for value in values[segment_slice]))
        return len(set(sums)) <= 1

    return digit_predicate_core(
        position_vars,
        is_valid,
        sudokuNum=sudokuNum,
        name="region_sum_direct",
        coreType=coreType,
        max_dense_cells=max_dense_cells,
    )

