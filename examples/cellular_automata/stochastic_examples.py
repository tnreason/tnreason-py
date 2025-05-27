from examples.cellular_automata import structure as st

from tnreason import application

"""
Provides templates for soft-logic cellular automata
"""


def get_random_inferer(blackweight=0):
    return application.InferenceProvider(
        application.HybridKnowledgeBase(
            weightedFormulas={"new": ["n", blackweight]},
        )
    )


def get_precessor_influenced_random_inferer(directWeight=0.1, neighborWeight=0.1):
    return application.InferenceProvider(
        application.HybridKnowledgeBase(
            weightedFormulas={"before": ["eq", "b", "n", directWeight],
                              "minus1": ["eq", "m1", "n", neighborWeight],
                              "plus1": ["eq", "p1", "n", neighborWeight]},
        )
    )


if __name__ == "__main__":
    basePath = "/Users/alexgoessmann/Documents/ENEXA/tnreason/version1/examples/cellular_automata/example_plots/"

    st.visualize_evolution(st.compute_evolution([0 for i in range(201)], get_random_inferer(), evolNum=100),
                           storePath=basePath + "random.png")

    dWeight = 1
    nWeight = 0.5
    st.visualize_evolution(st.compute_evolution([0 for i in range(201)],
                                                get_precessor_influenced_random_inferer(dWeight, nWeight), evolNum=100),
                           storePath=basePath + "precessor_influenced" + str(dWeight) + "_" + str(nWeight) + ".png")
