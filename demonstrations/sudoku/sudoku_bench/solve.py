from tnreason import reasoning

from demonstrations.sudoku import representation as rep
from demonstrations.sudoku import solution as sol
from demonstrations.sudoku import visualization as vis

num = 3

import pandas as pd
import numpy as np

import time

from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence

df = pd.read_parquet('sample_data/nikoli100.parquet')

def solve_with_messageLimit(puzzleNum, messageLimit, visualization=False, verbose=False):

    evidenceDict = initial_board_into_evidence(df["initial_board"][puzzleNum])
    if verbose:
        print("Known atoms before message passing: {}".format(len(evidenceDict)))

    ## Representation ##

    caNetwork = rep.get_assignment_as_CANetwork(num=num, startAssignment=evidenceDict)

    ## Reasoning ##

    propagator = reasoning.get_inferer("ExpectationPropagator")(
        caNetwork=caNetwork,
        clusterDict=sol.get_simple_clusterDict(num=num),
        meanParamDict=dict()
    )
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

def batch_solution(messageLimit=100000, untilPuzzle=100):
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
    print("### In total: ### \n{} out of {} solved \nAverage time: {} seconds".format(successCount, untilPuzzle, int(totalTime/untilPuzzle)))
    return successes, exeTimes, startAssCounts, solutionAssCounts

if __name__ == "__main__":
    mLimit = 100000
    untilP = 100

    successes, exeTimes, startAssCounts, solutionAssCounts = batch_solution(messageLimit=mLimit, untilPuzzle=untilP)
    results = pd.DataFrame(data = {
        "Success": successes,
        "ExecutionTime": exeTimes,
        "Start Assignments": startAssCounts,
        "Found Assignments": solutionAssCounts
    })

    output_path = 'results/result_nikoli100_simpleMP_mLimit{}.csv'.format(mLimit)  # Specify your desired file path
    results.to_csv(output_path, index=False)