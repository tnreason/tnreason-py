from tnreason import engine
import pandas as pd
from experiments.constraint_networks.sudoku_tests import visualization as vis


def initial_board_into_evidence(boardString, colNum=3, rowNum=3):
    """
    Turns the board string in the nicoli_100 DataFrame into the evidence format.
    """
    evidenceDict = {}
    for i in range(len(boardString)):
        if boardString[i] != ".":
            # Calculate the column and row indices
            coarseColPos = i % colNum ** 2
            c1 = coarseColPos // colNum
            c2 = coarseColPos % colNum

            coarseRowPos = i // colNum ** 2
            r1 = coarseRowPos // rowNum
            r2 = coarseRowPos % rowNum

            evidenceDict[
                "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(int(boardString[i]) - 1)] = 1

    return evidenceDict


def evidence_to_constraints(evidenceDict):
    """
    Prepares truth basis vectors for known atomic variables (could equivalently intialize other such as position variables)
    """
    return {atomKey: engine.create_from_slice_iterator(shape=[2], colors=[atomKey],
                                                       sliceIterator=[(1, {atomKey: evidenceDict[atomKey]})]) for
            atomKey in evidenceDict}


def read_evidence(puzzleNum, filePath='sample_data/nikoli100.parquet'):
    df = pd.read_parquet(filePath)
    return evidence_to_constraints(initial_board_into_evidence(df["initial_board"][puzzleNum]))


def get_solution_array(puzzleNum, filePath='sample_data/nikoli100.parquet'):
    df = pd.read_parquet(filePath)
    return vis.evidence_to_array(initial_board_into_evidence(df["solution"][puzzleNum]))


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_parquet('sample_data/nikoli100.parquet')
    print(evidence_to_constraints(initial_board_into_evidence(df["initial_board"][1])))

    from experiments.constraint_networks.sudoku_tests import visualization as vis
    vis.visualize_sudoku(vis.evidence_to_array(initial_board_into_evidence(df["initial_board"][1])))
    vis.visualize_sudoku(vis.evidence_to_array(initial_board_into_evidence(df["solution"][0])))
