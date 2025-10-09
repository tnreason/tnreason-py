from tnreason import representation, application, reasoning, engine

from demonstrations.sudoku import sudoku_constraints as rep
from demonstrations.sudoku import sudoku_forward_mp as sol
from demonstrations.sudoku import visualization as vis
from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence
from demonstrations.sudoku.sudoku_bench.human_to_tnreason import translate_edges

num = 3

import pandas as pd
import os, sys, json
import time


from experiments.backtracking_search.check_caNetwork import check_consistency
from tnreason.representation.ca_network import featureDict_from_computationCoreDict

from pathlib import Path
HERE = Path(__file__).resolve().parent

df = pd.read_csv(HERE/'sample_data/challenge100.csv')

def get_open_ai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required to translate rules with gpt5-nano.")
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"openai package is required: {e}")
    return OpenAI(api_key=key)

def get_sakana_constraints_openai(extraConstraints, num=3):
    """
    The additional Sakana constraints for Sudoku puzzles.
    """
    # client = get_open_ai_client()
    # # system = (
    # #     'Translate Sudoku variant rules into logical constraints on the Sudoku grid. '
    # #     'In the tnreason language the Sudoku variables have 5 indices: block number for the row, the row in that block, block number for the column, the column in that block, the possible value (1,...,9).' \
    # #     'To check out how it is done for the normal Sudoku rules, I attach the code for that here:' \
    # #     '- In each column each number appears exactly once. leads to {"col_" + str(n) + "_" + str(c1) + "_" + str(c2): ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) for r1 in range(num) for r2 in range(num)] for c1 in range(num) for c2 in range(num) for n in range(num ** 2)}' \
    # #     '- In each row each number appears exactly once. leads to {"row_" + str(n) + "_" + str(r1) + "_" + str(r2): ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) for c1 in range(num) for c2 in range(num)] for r1 in range(num) for r2 in range(num) for n in range(num ** 2)}' \
    # #     '- In each square each number appears exactly once. leads to {"square_" + str(n) + "_" + str(r1) + "_" + str(c1): ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) for r2 in range(num) for c2 in range(num)] for r1 in range(num) for c1 in range(num) for n in range(num ** 2)}'
    # #     '- Exactly one number is assigned at each position. leads to {"pos_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2): ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n) for n in range(num ** 2)] for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num)}'
    # #     'Keep the code similar to that notation.' \
    # #     )
    # system = (
    #     'You are an expert in translating Sudoku variant rules into logical constraints on the Sudoku grid. ' \
    #     'In the puzzle the sudoku is encoded by variables of the form r1c6, endoing the row and column (here row 1 and column 6), which take values in 1,2,3,4,5,6,7,8,9.' \
    #     'This is translated to variables of the form a_0_0_1_2_n, where the first two indices are the block row and the row in that block (here block row 0 and row 0 in that block), the second two indices are the block column and the column in that block (here block column 1 and column 2 in that block), and the last index is the number n (here n varies between 0 and 8, encoding numbers 1 to 9).' \
    #     'Translate the given rules to logical constraints in the form of slice ierators, e.g. of the form (1, {u: i, v: j}), which means that it is possible that u has value i and v has value j according to the given constraint.' \
    #     'It is possible to connect more variables like this, e.g. (1, {u: i, v: j, w: k}), meaning that u has value i, v has value j and w has value k.' \
    #     'Finally, the constraints should be collected in a list. Each list entry is a dictionary with the keys "shapes", "variables", "name", "slice_iterator":'
    #     '- The shape is the size of the tensor connecting the vriables. So the size of the connected variables multiplied.' \
    #     '- The variables entry is a dictionary of the variables connected in this constraint.'
    #     '- The name is the name of the edge.' \
    #     '- The slice iterator tuples of the form above describe the possibilities according to the constraints.' \
    #     "Only provide the list of dictionaries, no explanations."
    # )
    # rules_raw = extraConstraints["rules"]
    # # split into non-empty sentences robustly
    # rules = [s.strip() for s in rules_raw.split('.') if s.strip()]
    # user = (
    #         "Rules:\n" + "\n".join(f"- {r}." for r in rules) +
    #         "\nVisual elements:\n" + extraConstraints.get("visual_elements","") +
    #         "\nEncoded puzzle:\n" + extraConstraints.get("encoded_puzzle","")
    #         # + "\nHere the Sudoku variables are indexed by the row and the column and the numbers vary between 1 and 9" \
    #         # + "\nTranslate these rules into logical constraints in the tnreason language on the Sudoku grid, similar to the examples I provided." \
    # )
    
    # print("user:", user)

    # # sys.exit(0)

    # r = client.responses.create(
    #     model="gpt-5-nano",
    #     input=[{"role":"system","content":system},
    #            {"role":"user","content":user}]
    # )
    # print("response:", r.output_text)
    # txt = r.output_text.strip() 
    # ir = json.loads(txt) 
    path = HERE/'data.json'
    # with path.open('w', encoding="utf-8") as f:
    #     json.dump(ir, f, indent=2, ensure_ascii=False)
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
        d = translate_edges(d)
    # print("Chatty: ",d)
    return d

def get_sakana_constraints_public_api(extraConstraints, num=3):
    return

def get_assignment_as_CANetwork(num, startAssignment, extraConstraints):
    """
    Prepares the Sudoku game with a start assignment as a Computation-Activation Network
    """
    constraints = rep.get_sudoku_constraints(num)

    atomVariables = ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
                     for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num) for n in
                     range(num ** 2)]
    
    sakana_constraints = get_sakana_constraints_openai(extraConstraints, num)
    sakana_constraints = {key: 
        engine.create_from_slice_iterator(
        shape=sakana_constraints[key]['shapes'],
        colors=sakana_constraints[key]["variables"],
        sliceIterator=sakana_constraints[key]["slice_iterator"],
        name=f"{sakana_constraints[key]["name"]}",
        ) for key in sakana_constraints.keys()}
    sakana_features = featureDict_from_computationCoreDict(sakana_constraints)
    
    # constraints = {**constraints, **sakana_constraints}
    computationCores = {**sakana_constraints, **application.create_categorical_cores(constraints)}

    return representation.ComputationActivationNetwork(
        featureDict={
            **{atomKey: representation.HardPartitionFeature(featureColors=[atomKey], affectedComputationCores={}) for
               atomKey in atomVariables},  # Atomic Variable Features
            **{categoricalKey: representation.HardPartitionFeature(featureColors=[categoricalKey],
                                                                   affectedComputationCores={},
                                                                   shape=[num ** 2])
               for categoricalKey in constraints},
            **{categoricalKey + "_" + atomKey: representation.PassiveFeature(
                featureColors=[], shape=[], affectedComputationCores=[
                    categoricalKey + "_" + atomKey + representation.suf.atoCoreSuf])
               for categoricalKey in constraints for atomKey in constraints[categoricalKey]},
            **sakana_features
        },
        computationCoreDict=computationCores,
        canParamDict={evidenceKey: representation.create_basis_core(name=evidenceKey, shape=[2],
                                                                    colors=[evidenceKey],
                                                                    numberTuple=[startAssignment[evidenceKey]])
                      for evidenceKey in startAssignment}
    )

def get_forward_mp_inferer(num, startAssignment,extraConstraints):
    constraints = rep.get_sudoku_constraints(num)
    atomVariables = ["a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
                     for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num) for n in
                     range(num ** 2)]
    
    testAssignment =  {"a_2_0_0_0_0": 1,"a_2_1_0_1_0": 1,"a_2_2_0_2_0": 1}
    caNet = get_assignment_as_CANetwork(num=num, startAssignment=testAssignment, extraConstraints=extraConstraints)
    print("check consisteny",check_consistency(caNet, testAssignment))

    return reasoning.ForwardMessagePasser(
        caNetwork=caNet,
        messageClusters={
            **{atomKey: [atomKey] for atomKey in atomVariables},
            **{categoricalKey: [categoricalKey] for categoricalKey in constraints}
        },
        inferenceClusters={
            categoricalKey + "_" + atomKey: [categoricalKey, atomKey,
                                             categoricalKey + "_" + atomKey]
            for categoricalKey in constraints for atomKey in constraints[categoricalKey]
        },
        meanParamDict={
            **{atomKey: engine.create_from_slice_iterator(
                shape=[2], colors=[atomKey],
                sliceIterator=[(1, {})]) for atomKey in atomVariables}},
        **{categoricalKey: engine.create_from_slice_iterator(
            shape=[num * 2], colors=[categoricalKey],
            sliceIterator=[(1, {})]) for categoricalKey in constraints}
        )

def solve_with_messageLimit(puzzleNum, messageLimit, visualization=False, verbose=True):

    evidenceDict = initial_board_into_evidence(df["initial_board"][puzzleNum])
    if verbose:
        print("Known atoms before message passing: {}".format(len(evidenceDict)))

    extraDict = {"rules": df["rules"][puzzleNum],
                 "visual_elements": df["visual_elements"][puzzleNum],
                 "encoded_puzzle": df["encoded_puzzle"][puzzleNum]}

    ## Reasoning ##
    propagator = get_forward_mp_inferer(num, evidenceDict,extraDict)
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
    untilP = 1

    solve_with_messageLimit(29, messageLimit=mLimit)


    
    # successes, exeTimes, startAssCounts, solutionAssCounts = batch_solution(messageLimit=mLimit, untilPuzzle=untilP)
    # results = pd.DataFrame(data = {
    #     "Success": successes,
    #     "ExecutionTime": exeTimes,
    #     "Start Assignments": startAssCounts,
    #     "Found Assignments": solutionAssCounts
    # })
    # print(results)
    # output_path = 'results/result_nikoli100_simpleMP_mLimit{}.csv'.format(mLimit)  # Specify your desired file path
    # #results.to_csv(output_path, index=False)
