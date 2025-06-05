

#from tnreason.application.storage import save_as_yaml, load_from_yaml

from tnreason.representation import suffixes as suf

from tnreason.representation.cnf_to_cores import weightedFormulas_to_sparseCore

from tnreason.representation.auxiliary_cores import create_boolean_head

from tnreason.representation.coordinate_calculus import coordinatewise_transform, create_tensor_encoding, \
    create_trivial_core, create_basis_core, create_vanishing_core

from tnreason.representation.basis_calculus import create_relational_encoding, create_partitioned_relational_encoding, \
    create_interpretation_vector

from tnreason.representation.features import ComputationActivationNetwork, SingleSoftFeature, SoftPartitionFeature, \
    HardPartitionFeature, PassiveFeature

#from tnreason.application.script_transform import get_all_atom_colors, get_atom_colors, \
#    drop_color_suffixes_from_assignment, create_solution_expression, add_color_suffixes

#from tnreason.application.data_to_cores import create_data_cores

#from tnreason.application.formulas_to_cores import create_formula_computation_cores, \
#    create_formulas_cores, create_expressionDict_computation_cores, get_formula_color, \
#    create_atom_evidence_cores

#from tnreason.application.categoricals_to_cores import create_categorical_cores, create_atomization_cores, \
#    create_constraintCoresDict