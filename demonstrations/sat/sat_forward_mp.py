from demonstrations.sat import sat_constraints as satc
from tnreason.reasoning import message_passing as mp

from tnreason import representation, engine

import numpy as np

def get_forward_mp_inferer(clauseList):
    atomKeys = {key for clause in clauseList for key in clause.keys()}

    messageClusters = {atomKey + "_cluster": ["atomFeature_"+ str(atomKey)] for atomKey in atomKeys}
    inferenceClusters = {
        "clauseCluster_"+str(i): ["clauseFeature_"+str(i)] + ["atomFeature_"+str(atomKey) for atomKey in clauseDict] for
        i, clauseDict in enumerate(clauseList)}
    return mp.ForwardMessagePasser(
        caNetwork=satc.get_clauseList_as_CANetwork(clauseList),
        messageClusters=messageClusters,
        meanParamDict={
            atomKey: engine.create_from_slice_iterator(
                shape=[2], colors=[atomKey],
                sliceIterator=[(1, {})]) for atomKey in atomKeys},
        inferenceClusters=inferenceClusters
    )

def retrieve_atoms(mpInferer):
    assignment = {}
    for featureKey in mpInferer.caNetwork.featureDict:
        if featureKey.startswith("atomFeature_"):
            possibilitiesCore = engine.convert(mpInferer.caNetwork.canParamDict[featureKey], outCoreType="NumpyCore")
            if engine.contract(coreDict={"_": possibilitiesCore},
                               openColors=[])[:] == 1:
                assignment[featureKey] = np.where(possibilitiesCore.values != 0)[0][0]
    return assignment