import itertools
from typing import Dict
from tnreason import engine, reasoning, representation
from copy import deepcopy
from tnreason.representation import feature_naming

def reinitiallize_canetwork_with_new_inference_clusters(ca_network, new_features):
    """
    Re-initialize the CANetwork with new inference clusters. This is a more expensive way to add inference clusters, but can be useful for testing and debugging.
    """
    print("feautureDict before:", ca_network.featureDict['row_0_0_0_a_0_0_1_1_0'])
    print("computationCoreDict before:", ca_network.computationCoreDict['col_0_0_0_a_0_0_0_0_0_atoC'].__class__, ca_network.computationCoreDict['col_0_0_0_a_0_0_0_0_0_atoC'].shape)
    print("canParamDict before:",list(ca_network.canParamDict.keys())[0], ca_network.canParamDict[list(ca_network.canParamDict.keys())[0]], ca_network.canParamDict['col_5_1_2_a_0_1_1_2_5'].__class__)
    caNet = representation.ComputationActivationNetwork(
        featureDict={**ca_network.featureDict, **new_features},
        computationCoreDict=ca_network.computationCoreDict,
        canParamDict=ca_network.canParamDict
    )
    exit()
    return caNet


# def get_assignment_as_CANetwork(num, startAssignment):
#     """
#     Prepares the Sudoku game with a start assignment as a Computation-Activation Network
#     """
#     constraints = get_sudoku_constraints(num)

#     comCoreDict = application.create_categorical_cores(constraints)
#     canParamDict = {evidenceKey: representation.create_basis_core(name=evidenceKey, shape=[2],
#                                                                     colors=[evidenceKey],
#                                                                     numberTuple=[startAssignment[evidenceKey]])
#                       for evidenceKey in startAssignment}
#     featureDict = representation.standard_featureDict_from_computationCoreDict(comCoreDict)
    


def _initial_mean_params_from_canetwork(ca_network) -> Dict[str, object]:
    mean_params = {}
    for feature_key, feature in ca_network.featureDict.items():
        if "passive" in feature.featureProperties:
            continue
        mean_params[feature_key] = engine.create_from_slice_iterator(
            shape=feature.shape,
            colors=feature.featureColors,
            sliceIterator=[(1, {})],
        )
    return mean_params

def _mean_params_to_atom_evidence(mean_param_dict):
    """
    Extract atom assignments a_* = 1 from mean parameters.
    """
    return {
        feature_key: 1
        for feature_key in mean_param_dict
        if feature_key.startswith("a_") and mean_param_dict[feature_key][{feature_key: 0}] == 0
    }

def visualize_propagator_assignment(propagator, num=3, label="Current Assignment", path=None):
    """
    Visualize the current Sudoku assignment implied by propagator.meanParamDict.
    Returns the extracted evidence dict so callers can compare steps.
    """
    from demonstrations.sudoku import visualization as vis

    evidence = _mean_params_to_atom_evidence(propagator.meanParamDict)
    array = vis.evidence_to_array(evidence, num=num)
    vis.visualize_sudoku(array.astype(int), number=num, label=label, path=path)
    return evidence

def eval_inferenceCluster(propagator, new_info, return_bool=False, maxMessageCount=None, verbose=False):
    """
    input:
    - propagator: the propagator to perform inference
    - new_info: a list of lists of variable names, e.g. [[var1,var2], [var1,var3,var4]]. Each list of variable names corresponds to a new inference cluster.
    - maxMessageCount: an optional maximum number of messages to pass before stopping propagation and returning
    Run message passing to convergence and return the number of changed feature means or a boolean if any features were changed. (Boolean is quicker to compute, but less informative. It only runs message passing until the first change)
    """
    nonTrivialFeatureKeys = []
    new_inference_clusters = {}
    for vars in new_info:
        new_cluster = [feature_naming([var1,var2]) for var1, var2 in itertools.combinations_with_replacement(vars, 2) if var1 != var2 and feature_naming([var1,var2]) in propagator.caNetwork.featureDict]
        print("new cluster:", vars + new_cluster)
        new_inference_clusters[feature_naming(vars[::-1])] = vars+new_cluster #+ ["_".join(vars[::-1])]
        nonTrivialFeatureKeys += vars+new_cluster

    inference_clusters = {**propagator.inferenceClusters, **new_inference_clusters}
    propagator.update_inferenceClusters(inference_clusters)
    if return_bool:
        return propagate_until_any_variable_change(propagator, nonTrivialFeatureKeys, maxMessageCount=maxMessageCount, verbose=verbose), propagator
    else:
        # run message passing to convergence
        return propagate_until_convergence_count_variable_changes(propagator, nonTrivialFeatureKeys, maxMessageCount=maxMessageCount, verbose=verbose), propagator

def propagate_until_any_variable_change(propagator, nonTrivialFeatureKeys, maxMessageCount=None, verbose=False):
    """
    Run message passing and return True as soon as a feature mean changes.
    Return False when propagation converges without any changed feature means.
    """
    propagator.add_affected_directions(nonTrivialFeatureKeys)
    start_message_count = propagator.messageCount

    while len(propagator.messageQueue) > 0 and (maxMessageCount is None or propagator.messageCount < start_message_count + maxMessageCount):
        sendCluster, receiveCluster = propagator.messageQueue.pop()
        changedMeans = propagator.compute_canParam_message(sendCluster, receiveCluster)
        if verbose:
            print("Message {} passed from cluster {} to cluster {}. Changed feature means: {}".format(
                    propagator.messageCount, sendCluster, receiveCluster, changedMeans))

        propagator.add_affected_directions( [featureKey for featureKey in changedMeans],
                                            exceptionList=[(sendCluster, receiveCluster)],verbose=verbose)
        if getattr(propagator, "cleanQueue", True):
            propagator.remove_fixed_messages()
        if changedMeans:
            return True
    return False

def propagate_until_convergence_count_variable_changes(propagator, nonTrivialFeatureKeys, maxMessageCount=None, verbose=False):
    """
    Run message passing to convergence and return how many variable-change events occurred.
    """
    propagator.add_affected_directions(nonTrivialFeatureKeys)
    start_message_count = propagator.messageCount
    variable_change_count = 0

    while len(propagator.messageQueue) > 0 and (maxMessageCount is None or propagator.messageCount < start_message_count + maxMessageCount):
        sendCluster, receiveCluster = propagator.messageQueue.pop()
        old_propagator = deepcopy(propagator)
        changedMeans = propagator.compute_canParam_message(sendCluster, receiveCluster)
        variable_change_count += len(changedMeans)

        ######## for debugging 
        if verbose:
            print("Message {} passed from cluster {} to cluster {}. Changed feature means: {}".format(
                    propagator.messageCount, sendCluster, receiveCluster, changedMeans))
            if changedMeans:
                print("Current mean params for changed features:")
                for featureKey in changedMeans:
                    colors = propagator.meanParamDict[featureKey].colors
                    length = propagator.meanParamDict[featureKey].shape[0]
                    print(featureKey, propagator.meanParamDict[featureKey].shape, [ old_propagator.meanParamDict[featureKey][{featureKey: i }] for i in range(length)])
                    print(featureKey,  [ propagator.meanParamDict[featureKey][{featureKey: i }] for i in range(length)])
                    for color in colors:
                        length = propagator.meanParamDict[featureKey].shape[colors.index(color)]
                        print(featureKey, color,  [ old_propagator.meanParamDict[color][{color: i }] for i in range(length)])
                        print(featureKey, color,  [ propagator.meanParamDict[color][{color: i }] for i in range(length)])
        ########
        
        propagator.add_affected_directions([featureKey for featureKey in changedMeans],
                                            exceptionList=[(sendCluster, receiveCluster)],verbose=verbose)
        if getattr(propagator, "cleanQueue", True):
            propagator.remove_fixed_messages()
    return variable_change_count

if __name__ == "__main__":
    # This is a demonstration of how to measure effect of inference clusters for a standard sudoku example.

    # Assign a standard Sudoku problem as a CANetwork
    from demonstrations.sudoku.examples import standard_sudoku
    from demonstrations.sudoku import visualization as vis

    def rc_to_atom_assignment(r, c, n):
        r1 = r // 3
        r2 = r % 3
        c1 = c // 3
        c2 = c % 3
        return "a_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2) + "_" + str(n)
    evidenceDict = {
                    rc_to_atom_assignment(3, 0, 3): 1,
                    rc_to_atom_assignment(4, 0, 4): 1,
                    rc_to_atom_assignment(3, 1, 1): 1,
                    rc_to_atom_assignment(4, 1, 2): 1,
                    rc_to_atom_assignment(5, 3, 0): 1,
                    rc_to_atom_assignment(6, 4, 0): 1,
                    rc_to_atom_assignment(7, 6, 0): 1,
                    rc_to_atom_assignment(8, 0, 1): 1,
                    }
    # Construct the CANetwork and visualize the initial assignment
    ca_network = standard_sudoku.get_assignment_as_CANetwork(3,evidenceDict)
    start_array = vis.evidence_to_array(evidenceDict, num=3)
    vis.visualize_sudoku(start_array.astype(int), number=3, label="Start Assignment", path = "experiments/alphasudoku/start_assignment.png")
    
    # Construct propagator
    message_clusters, inference_clusters = reasoning.standard_clusters_from_computationCoreDict(ca_network.computationCoreDict)
    propagator = reasoning.ForwardMessagePasser(
                        caNetwork=ca_network,
                        messageClusters=message_clusters,
                        inferenceClusters=inference_clusters,
                        meanParamDict=_initial_mean_params_from_canetwork(ca_network),
                        allowClearning=False,
                    )
    # Propagate already known assignments with already known rules
    nonTrivialFeatureKeys = evidenceDict.keys()
    propagator.propagate_until_convergence(nonTrivialFeatureKeys=nonTrivialFeatureKeys)
    visualize_propagator_assignment(propagator, label="After Initial Propagation", path="experiments/alphasudoku/after_initial_propagation.png")


    # Standard Sudoku variable names:
    # - pos_r1_r2_c1_c2 for cell position variables (shape: 9)
    # - a_r1_r2_c1_c2_n for atom variables indicating whether digit n is assigned to cell (r1,r2,c1,c2) (shape: 2)
    # - row_n_r1_r2 for row constraints indicating where digit n is assigned in row (r1,r2) (shape: 9)
    # - col_n_c1_c2 for column constraints indicating where digit n is assigned in column (c1,c2) (shape: 9)
    # - square_n_r1_c1 for square constraints indicating where digit n is assigned in square (r1,c1) (shape: 9)

    ################################### Example inference cluster: This should give us new information
    # Initially, there are two possible positions for the number 1 in the square 2_0, i.e. "square_0_2_0" = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0] ("square_n_c1_c2")
    # This inference cluster leads to the assignment of 1 to a specific cell with "square_0_2_0" = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    # The list only contains the variables, that should be included in the cluster. The test automatically looks for features that connect pairs of the variables.
    new_info = [["col_0_0_2","square_0_1_0", rc_to_atom_assignment(3, 2, 0), rc_to_atom_assignment(4,2,0)]]

    # We measure the effect of the inference cluster with the number of changed means
    print("Before new information: square", [ propagator.meanParamDict["square_0_2_0"][{"square_0_2_0": i }] for i in range(9)])
    change_count, _ = eval_inferenceCluster(deepcopy(propagator), new_info, return_bool=False, maxMessageCount=5000, verbose=False)
    print("After new information: square", [ propagator.meanParamDict["square_0_2_0"][{"square_0_2_0": i }] for i in range(9)])
    print(f"Any variable changed: {change_count}")
    visualize_propagator_assignment(propagator, label="After Cluster Propagation", path="experiments/alphasudoku/after_cluster_propagation.png")

    # We just want to know if the inference cluster had any effect, without waiting for full convergence, so we run message passing only until the first change occurs
    any_change, _ = eval_inferenceCluster(deepcopy(propagator), new_info, return_bool=True, maxMessageCount=5000, verbose=False)
    print(f"Variable change count: {any_change}")

    ################################## Example inference cluster: This should not give us new information
    new_info = [["a_0_0_0_0_0", "a_0_0_1_1_0", "pos_1_1_1_1"]]
    any_change, _ = eval_inferenceCluster(deepcopy(propagator), new_info, return_bool=True, maxMessageCount=5000, verbose=False)
    print(f"Any variable changed: {any_change}")
    change_count, _ = eval_inferenceCluster(deepcopy(propagator), new_info, return_bool=False, maxMessageCount=5000, verbose=False)
    print(f"Variable change count: {change_count}")
