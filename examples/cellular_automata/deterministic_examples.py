from examples.cellular_automata import structure as st

from tnreason import knowledge

"""
Provides templates for hard-logic deterministic cellular automata
"""

def get_ternary_inferer(ruleNumber):
    return get_equivalence_inferer(["t"+str(ruleNumber), "m1", "b", "p1"])


def get_equivalence_inferer(formula):
    return knowledge.InferenceProvider(
        knowledge.HybridKnowledgeBase(
            facts={"f1": ["eq", "n", formula]}
        )
    )


def get_right_propagation():
    return get_equivalence_inferer("m1")


def get_chess_propagation():
    return get_equivalence_inferer(["or", "m1", "p1"])


def get_wave_propagation():
    return get_equivalence_inferer(["not", "b"])


def get_one_hot_start(dim=10, pos=3):
    return [1 if i == pos else 0 for i in range(dim)]


if __name__ == "__main__":
    basePath = "/Users/alexgoessmann/Documents/ENEXA/tnreason/version1/examples/cellular_automata/example_plots/"

    st.visualize_evolution(st.compute_evolution([0 for i in range(201)], get_ternary_inferer(30), evolNum=100),
                           storePath=basePath + "rule30.png")
    exit()
    st.visualize_evolution(st.compute_evolution([0 for i in range(201)], get_wave_propagation(), evolNum=100),
                           storePath=basePath + "wave_(not_b).png")
    st.visualize_evolution(st.compute_evolution(get_one_hot_start(201, 100), get_right_propagation(), evolNum=100),
                           storePath=basePath + "right_m1.png")
    st.visualize_evolution(st.compute_evolution(get_one_hot_start(201, 100), get_chess_propagation(), evolNum=100),
                           storePath=basePath + "chess_(or_m1_p1).png")
