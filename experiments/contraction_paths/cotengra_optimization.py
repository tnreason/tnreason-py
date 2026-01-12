def extract_contraction_specs(coreDict):
    """
    Given a coreDict to be contracted, extracts the core colors and their dimensions.
    """
    inputs = [coreDict[coreKey].colors for coreKey in coreDict]
    sizeDict = dict()
    for coreKey in coreDict:
        for i, color in enumerate(coreDict[coreKey].colors):
            if color not in sizeDict:
                sizeDict[color] = coreDict[coreKey].shape[i]
    return inputs, sizeDict


if __name__ == "__main__":
    import cotengra as ctg
    from tnreason import engine
    import numpy as np

    coreDict = {
        "c1": engine.get_core("NumpyCore")(
            values=np.random.rand(2, 2),
            colors=["red", "blue"]
        ),
        "c2": engine.get_core("NumpyCore")(
            values=np.random.rand(2, 2),
            colors=["sledz", "blue"]
        ),
        "c3": engine.get_core("NumpyCore")(
            values=np.random.rand(2, 2),
            colors=["sledz", "blue"]
        )
    }

    inputs, sizeDict = extract_contraction_specs(coreDict)
    output = ["sledz", "blue"]

    opt = ctg.HyperOptimizer()
    tree = opt.search(inputs, output, sizeDict)
    #    tree.plot_flat()
    print(tree.contraction_width(), tree.contraction_cost())
    print(tree)

    path = tree.get_path()
    print(path)

    np.einsum("rb,sb,sb->sb",
              [np.random.rand(2, 2), np.random.rand(2, 2)],
              optimize=path)
