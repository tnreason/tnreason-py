from demonstrations.sudoku import sudoku_constraints as rep
from demonstrations.sudoku import sudoku_forward_mp as sol
from demonstrations.sudoku import visualization as vis

num = 3

import pandas as pd

import time

from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence

df = pd.read_parquet('sample_data/nikoli100.parquet')


def solve_with_messageLimit(puzzleNum, messageLimit, visualization=False, verbose=False):
    evidenceDict = initial_board_into_evidence(df["initial_board"][puzzleNum])
    if verbose:
        print("Known atoms before message passing: {}".format(len(evidenceDict)))

    propagator = sol.get_forward_mp_inferer(num, evidenceDict)
    startTime = time.time()
    propagator.propagate_until_convergence(evidenceDict.keys(), maxMessageCount=messageLimit, verbose=False)
    execTime = time.time() - startTime
    if verbose:
        print("Propagation with message limit {} took {} seconds.".format(messageLimit, int(execTime)))

    solutionEvidence = sol.meanParams_to_evidence(propagator.meanParamDict)
    if verbose:
        print("Known atoms after message passing: {}".format(len(solutionEvidence)))

    trueSolution = initial_board_into_evidence(df["solution"][puzzleNum])
    for atom in solutionEvidence:
        assert atom in trueSolution

    if visualization:
        start_array = vis.evidence_to_array(evidenceDict, num=num)
        vis.visualize_sudoku(start_array.astype(int), number=3, label="Start Assignment")
        solution_array = vis.evidence_to_array(solutionEvidence, num=num)
        vis.visualize_sudoku(solution_array.astype(int), number=3, label="Solution Assignment")

    return len(solutionEvidence), execTime, len(evidenceDict)


def batch_solution_mp(messageLimit=100000, untilPuzzle=100):
    successes = []
    exeTimes = []
    startAssCounts = []
    solutionAssCounts = []

    successCount = 0
    totalTime = 0
    for i in range(untilPuzzle):
        foundLen, exeTime, startAssCount = solve_with_messageLimit(i, messageLimit=messageLimit)
        print(i, foundLen, int(exeTime), startAssCount)
        if foundLen == 81:
            successCount += 1
            successes.append(True)
        else:
            successes.append(False)
        exeTimes.append(exeTime)
        startAssCounts.append(startAssCount)
        solutionAssCounts.append(foundLen)

        totalTime += exeTime
    print("### In total: ### \n{} out of {} solved \nAverage time: {} seconds".format(successCount, untilPuzzle,
                                                                                      int(totalTime / untilPuzzle)))
    return successes, exeTimes, startAssCounts, solutionAssCounts


def batch_solution_backtracking_mp(maxIterations=1000, untilPuzzle=100, verbose=False):
    from experiments.backtracking_search import backtrack_caNetwork as bt

    successes = []
    exeTimes = []
    startAssCounts = []
    solutionAssCounts = []

    successCount = 0
    totalTime = 0
    for puzzleNum in range(untilPuzzle):

        evidenceDict = initial_board_into_evidence(df["initial_board"][puzzleNum])

        startAssCounts.append(len(evidenceDict))
        if verbose:
            print("Known atoms before message passing: {}".format(len(evidenceDict)))

        backtracker = bt.Backtracker(propagator=sol.get_forward_mp_inferer(num, evidenceDict), assignment=evidenceDict)

        ## Solution by backtracking search
        startTime = time.time()
        solutionAssignment, iterationCount = backtracker.search(maxIterations=maxIterations)
        print("Backtracking for puzzle {} took {} iterations.".format(puzzleNum, iterationCount))
        exeTime = time.time() - startTime
        solutionAssCounts.append(len(solutionAssignment))

        try:
            ## Check Solution
            trueSolution = initial_board_into_evidence(df["solution"][puzzleNum])
            for atom in solutionAssignment:
                assert atom in trueSolution

            ## Evaluation
            foundLen = len(solutionAssignment)
            if foundLen == 81:
                successCount += 1
                successes.append(True)
            else:
                successes.append(False)
            exeTimes.append(exeTime)

        except:
            print("## SOLUTION INCORRECT ##")
            successes.append(False)
            exeTimes.append(exeTime)

            vis.visualize_sudoku(vis.evidence_to_array(solutionAssignment, num=num).astype(int), number=3, label="Solution Assignment")
            #vis.visualize_sudoku(vis.evidence_to_array(initial_board_into_evidence(df["solution"][puzzleNum]), num=num).astype(int), number=3, label="True Solution")

        totalTime += exeTime
    print("### In total: ### \n{} out of {} solved \nAverage time: {} seconds".format(successCount, untilPuzzle,
                                                                                      int(totalTime / untilPuzzle)))
    return successes, exeTimes, startAssCounts, solutionAssCounts


if __name__ == "__main__":
    mLimit = 100000
    untilP = 10

    maxIterations = 200
    successes, exeTimes, startAssCounts, solutionAssCounts = batch_solution_backtracking_mp(maxIterations=maxIterations,
                                                                                            untilPuzzle=untilP)
    results = pd.DataFrame(data={
        "Success": successes,
        "ExecutionTime": exeTimes,
        "Start Assignments": startAssCounts,
        "Found Assignments": solutionAssCounts
    })
    print(results)
    output_path = 'results/result_nikoli100_backtrackingMP_maxIterations{}.csv'.format(
        maxIterations)  # Specify your desired file path
    results.to_csv(output_path, index=False)


    exit()
    successes, exeTimes, startAssCounts, solutionAssCounts = batch_solution_mp(messageLimit=mLimit, untilPuzzle=untilP)
    results = pd.DataFrame(data={
        "Success": successes,
        "ExecutionTime": exeTimes,
        "Start Assignments": startAssCounts,
        "Found Assignments": solutionAssCounts
    })
    print(results)
    output_path = 'results/result_nikoli100_simpleMP_mLimit{}.csv'.format(mLimit)  # Specify your desired file path
    # results.to_csv(output_path, index=False)
