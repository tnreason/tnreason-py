"""Raw ordered network data shared by learning and runtime selection."""
import numpy as np


def network_data(network):
    """Raw ordered tensor data, not neural features; store once per dataset."""
    colors = list(network.dimension_dict)
    idx = {color: i for i, color in enumerate(colors)}
    cores = []
    for key, core in network.coresDict.items():
        values = getattr(core, "values", None)
        support = values != 0 if isinstance(values, np.ndarray) else np.array([
            core[dict(zip(core.colors, coordinate))] != 0 for coordinate in np.ndindex(tuple(core.shape))
        ]).reshape(core.shape)
        cores.append(dict(key=key, colors=[idx[color] for color in core.colors],
                          shape=[int(dim) for dim in core.shape], values=support.astype(bool).ravel().tolist()))
    return dict(colors=colors, dimensions=[int(network.dimension_dict[color]) for color in colors], cores=cores)
