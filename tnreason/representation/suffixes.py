# Suffixes and Infixes to connect with the report notation:
# Tandem in macros/organization_macros.tex where the strings are macros

"""
Core suffixes
"""

# Main cores
comCoreSuf = "_cC" # computation core
actCoreSuf = "_aC" # activation Core

# Sum trick cores
atoCoreSuf = "_atoC" # for categorical constraint (simplifying contraction of computed core with true activation)
vselCoreSuf = "_vselC" # for variable selection (simplifying variable selection computation)

"""
Color suffixes
"""

disVarSuf = "_dV"#"_dV" # distributed variable X
comVarSuf = "_cV" # computed variable Y
selVarSuf = "_sV" # selection variable L
terVarSuf = "_tV" # term variable O

"""
Core and Color suffix refiners
"""

# Computation Core Refiners
selCoreIn = "_s" # Selection Core

# Activation Core Refiners
eviCoreIn = "_e" # Evidence

# Neuron representation
heaIn = "_h" # head of neuron
funIn = "_f" # (activation) function selection
posIn = "_p" # position argument selection

# Data representation
datIn = "_d"