from tnreason.representation.formulas_to_cores import create_formulas_cores, create_raw_formula_cores, get_formula_color, \
    create_formula_head, create_boolean_head, create_evidence_cores, create_atom_evidence_cores, get_atoms, get_all_atoms
from tnreason.representation.categoricals_to_cores import create_categorical_cores, create_atomization_cores, create_constraintCoresDict
from tnreason.representation.neurons_to_cores import create_neuron, create_architecture, find_atom_colors, find_selection_dimDict, \
    create_solution_expression, find_selection_colors, parse_neuronNameDict_to_neuronColorDict
from tnreason.representation.data_to_cores import create_data_cores

from tnreason.representation.storage import save_as_yaml, load_from_yaml

from tnreason.representation import suffixes as suf

from tnreason.representation.cnf_to_cores import weightedFormulas_to_sparseCore

from tnreason.representation.creation_handling import create_tensor_encoding, \
     create_relational_encoding, create_partitioned_relational_encoding, \
    coordinatewise_transform, create_trivial_core, create_basis_core, create_trivial_cores