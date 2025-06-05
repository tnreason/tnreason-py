# application/__init__.py

from tnreason.application.inductive import HybridLearner
from tnreason.application.deductive import InferenceProvider

from tnreason.application.distributions import HybridKnowledgeBase, get_empirical_distribution, MarkovNetwork, ProposalDistribution
from tnreason.application.grafting import Grafter
from tnreason.application.batch_evaluation import KnowledgePropagator

from tnreason.application.knowledge_visualization import visualize

from tnreason.application.storage import load_from_yaml, save_as_yaml

from tnreason.application.categoricals_to_cores import create_categorical_cores, create_atomization_cores, \
    create_constraintCoresDict

def load_kb_from_yaml(loadPath):
    kb = HybridKnowledgeBase()
    kb.from_yaml(loadPath)
    return kb


