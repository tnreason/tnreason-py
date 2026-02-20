def initial_board_into_evidence(boardString, colNum = 3, rowNum = 3):
    """
    Turns the board string into the evidence format.
    """
    evidenceDict = {}
    for i in range(len(boardString)):
        if boardString[i] != ".":

            # Calculuate the column and row indices
            coarseColPos = i % colNum**2
            c1 = coarseColPos // colNum
            c2 = coarseColPos % colNum

            coarseRowPos = i // colNum**2
            r1 = coarseRowPos // rowNum
            r2 = coarseRowPos % rowNum

            evidenceDict["a_" +str(r1)+"_"+str(r2)+"_"+str(c1)+"_"+str(c2)+"_"+str(int(boardString[i])-1)] = 1

    return evidenceDict


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet('sample_data/nikoli100.parquet')
    print(df["initial_board"].head())
    print(df["solution"].head())

    from demonstrations.sudoku import visualization as vis
    vis.visualize_sudoku(vis.evidence_to_array(initial_board_into_evidence(df["initial_board"][1])))
    vis.visualize_sudoku(vis.evidence_to_array(initial_board_into_evidence(df["solution"][0])))
