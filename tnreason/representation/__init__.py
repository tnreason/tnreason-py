from tnreason.representation import suffixes as suf

from tnreason.representation.coordinate_calculus import coordinatewise_transform, create_tensor_encoding, \
    create_trivial_core, create_basis_core, create_vanishing_core

from tnreason.representation.basis_calculus import create_basis_encoding_from_lambda, create_partitioned_basis_encoding, \
    create_interpretation_vector

from tnreason.representation.features import SingleSoftFeature, SoftPartitionFeature, \
    HardPartitionFeature, PassiveFeature, EnergyDictFeature, SingleHybridFeature

from tnreason.representation.ca_network import ComputationActivationNetwork

from tnreason.representation.basisPlus_calculus import get_boolean_computation_core, get_selection_augmented_boolean_computation_core
