from pyqubo import Binary


def core_to_pyqubo_hamiltonian(core):
    binariesColorDict = {color: Binary(color) for color in core.colors}
    hamiltonian = 0
    for val, positions in iter(core):
        hamiltonian = hamiltonian + create_potential(
            val, {key for key in positions if not positions[key]}, {key for key in positions if positions[key]},
            binariesColorDict
        )
    return hamiltonian

def create_potential(val, neg, pos, binariesDict):
    potential = 1
    for color in neg:
        potential = potential * (1 - binariesDict[color])
    for color in pos:
        potential = potential * binariesDict[color]
    return val * potential


if __name__ == "__main__":
    from tnreason.demonstrations.cnf_representation import cnf_to_cores as ctc

    polyCore = ctc.weightedFormulas_to_sparseCore({
        "w1": ["imp", "a", "b", 0.678],
        "w2": ["a", 0.34]
    }, coreType="PandasCore")
    hamiltonian = core_to_pyqubo_hamiltonian(polyCore)
    model = hamiltonian.compile()

    qubo, offset = model.to_qubo()
    print(qubo)
