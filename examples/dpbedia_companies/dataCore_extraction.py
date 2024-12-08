from tnreason.engine.creation_handling import core_to_relational_encoding

from tnreason.encoding import suffixes as suf

def get_dataCores(importanceQueryCore, atomQueryCoreDict=dict(), dataColor="j" + suf.dataVariableSuffix,
                  categoricalColors=[], coreType=None,
                  contractionMethod="PolynomialContractor"):
    """
    :importanceQueryCore: Tensor Core representing the evaluation of the importance query (before slice enumeration!)
    :atomQueryCoreDict: Dictionary of Tensor Cores representing the evaluation of the atom extraction queries
    :dataColor: Color of the entry enumeration in the importanceQueryCore, which will be interpreted as the data color
    :coreType: Type of the resulting data cores
    """
    importanceQueryCore.enumerate_slices(enumerationColor=dataColor)
    dataCores = {atomKey + suf.dataCoreSuffix: core_to_relational_encoding(
        core=engine.contract({"imCore" + suf.queryCoreSuffix: importanceQueryCore, atomKey: atomQueryCoreDict[atomKey]},
                             openColors=[dataColor], method=contractionMethod), headColor=atomKey,
        outCoreType=coreType)[0] for atomKey in atomQueryCoreDict}
    if not len(categoricalColors) == 0:
        dataCores["_".join([color for color in categoricalColors]) + suf.dataCoreSuffix] = engine.contract(
            {"imCore": importanceQueryCore}, openColors=[dataColor] + categoricalColors, method=contractionMethod)
    return dataCores