datPre = "_d" # Data (used for core and color refinement)

"""
Cores
"""

comCoreSuf = "_cC" # computation core
actCoreSuf = "_aC" # activation Core

# Computation Core Refiners
selCorePre = "_s" # Selection Core

# Activation Core Refiners
eviCorePre = "_e" # Evidence
domCorePre = "_d" # Domain

# Sum trick cores
atoCoreSuf = "_atoC" # for categorical constraint (simplifying contraction of computed core with true activation)
vselCoreSuf = "_vselC" # for variable selection (simplifying variable selection computation)

"""
Colors
"""

disVarSuf = ""#_dV" # distributed variable X
comVarSuf = "_cV" # computed variable Y
selVarSuf = "_sV" # selection variable L
terVarSuf = "_tV" # term variable O