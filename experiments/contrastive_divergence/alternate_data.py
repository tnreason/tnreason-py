if __name__ == "__main__":
    """
    Data transformation now at the SampleCoreBase (in startSlices -> data)
    """

    from tnreason.algorithms import energy_based_algorithms as eba

    from tnreason import knowledge, encoding, engine

    dist = knowledge.HybridKnowledgeBase(weightedFormulas={"w1": ["imp", "a", "b", 0.23]})

    sampler = eba.EnergyGibbsSampleCore(dist.get_energy_dict(), colors=dist.distributedVariables,
                                        dimDict={color: 2 for color in dist.distributedVariables},
                                        # startSliceIterator=iter([(1, {"a" + encoding.suf.disVarSuf: 1,
                                        #                 "b" + encoding.suf.disVarSuf: 0})]),
                                        temperatureList=[1 for i in range(5)],
                                        sampleNum=3
                                        )

    print(sampler.to_core("PandasCore").values)
    exit()
    core = engine.convert(iter(sampler), "PandasCore")
    for entry in core:
        print(entry)

    print("again")
    for entry in core:
        print(entry)


    sampler = eba.EnergyGibbsSampleCore(dist.get_energy_dict(), colors=dist.distributedVariables,
                                        temperatureList=[1 for i in range(5)],
                                        dimDict={color: 2 for color in dist.distributedVariables},
                                        startSlices=[(1,dict()), (1,dict())]  #core
                                        )
    print("redone")
    for sample in sampler:
        print(sample)

    for sample in sampler:
        print(sample)


#    print(engine.convert(iter(sampler), "PandasCore").values)
