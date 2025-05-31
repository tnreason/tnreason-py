from tnreason.representation.formulas_to_cores import create_formulas_cores, create_raw_formula_cores, \
    get_formula_color, \
    create_formula_head, create_evidence_cores, create_atom_evidence_cores
from tnreason.representation.categoricals_to_cores import create_categorical_cores, create_atomization_cores, \
    create_constraintCoresDict
from tnreason.representation.neurons_to_cores import create_neuron, create_architecture, find_atom_colors, \
    find_selection_dimDict, \
    find_selection_colors
from tnreason.representation.data_to_cores import create_data_cores

from tnreason.representation.storage import save_as_yaml, load_from_yaml

from tnreason.representation import suffixes as suf

from tnreason.representation.cnf_to_cores import weightedFormulas_to_sparseCore

from tnreason.representation.creation_handling import create_trivial_core, create_basis_core, create_trivial_cores, create_boolean_head, \
    create_activation_vector, create_vanishing_core


from tnreason.representation.script_transform import get_all_atom_colors, get_atom_colors, \
    drop_color_suffixes_from_assignment, create_solution_expression, add_color_suffixes

from tnreason.representation.coordinate_calculus import coordinatewise_transform, create_tensor_encoding

from tnreason.representation.basis_calculus import create_relational_encoding, create_partitioned_relational_encoding, \
    create_interpretation_vector