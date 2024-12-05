from tnreason.encoding.formulas_to_cores import create_formulas_cores, create_raw_formula_cores, get_formula_color, \
    create_formula_head, create_boolean_head, create_evidence_cores, create_atom_evidence_cores, get_atoms, get_all_atoms
from tnreason.encoding.categoricals_to_cores import create_categorical_cores, create_atomization_cores, create_constraintCoresDict
from tnreason.encoding.neurons_to_cores import create_neuron, create_architecture, find_atom_colors, find_selection_dimDict, \
    create_solution_expression, find_selection_colors, parse_neuronNameDict_to_neuronColorDict
from tnreason.encoding.data_to_cores import create_data_cores

from tnreason.encoding.storage import save_as_yaml, load_from_yaml

from tnreason.encoding import suffixes as suf