from typing import Dict

from tnreason import engine, reasoning


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


def get_forward_mp_inferer_from_canetwork(ca_network, allow_cleaning: bool = False):
    message_clusters, inference_clusters = reasoning.standard_clusters_from_computationCoreDict(
        ca_network.computationCoreDict
    )
    return reasoning.ForwardMessagePasser(
        caNetwork=ca_network,
        messageClusters=message_clusters,
        inferenceClusters=inference_clusters,
        meanParamDict=_initial_mean_params_from_canetwork(ca_network),
        allowClearning=allow_cleaning,
    )

