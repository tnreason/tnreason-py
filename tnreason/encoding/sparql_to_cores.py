from tnreason import engine
from tnreason.engine.creation_handling import core_to_relational_encoding

from tnreason.encoding import suffixes as suf

from SPARQLWrapper import SPARQLWrapper, JSON


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


def queries_to_polynomialCores(endpointString, queryDict, startInterpretationDict=dict()):
    polyCoresDict = dict()
    for queryKey in queryDict:
        core, startInterpretationDict = query_to_polynomialCore(endpointString, queryDict[queryKey],
                                                                startInterpretationDict,
                                                                executionProvider="SPARQLWrapper",
                                                                name=queryKey + suf.queryCoreSuffix)
        polyCoresDict[queryKey + suf.queryCoreSuffix] = core
    return polyCoresDict, startInterpretationDict


def query_to_polynomialCore(endpointString, queryString, startInterpretationDict=dict(),
                            executionProvider="SPARQLWrapper", name="ExampleQuery"):
    """
    Evaluates queries and outputs as
    Can integrate further executionProviders (rdflib or tentris)
    """
    if executionProvider is "SPARQLWrapper":
        querySolution = wrapper_json_evaluate_query(endpointString, queryString)
        entryPositionList, interpretationDict = query_result_to_entryPositionList(querySolution,
                                                                                  startInterpretationDict)
        variableNames = querySolution["head"]["vars"]
    else:
        raise ValueError("SPARQL Execution provider {} not supported!".format(executionProvider))

    return engine.get_core("PolynomialCore")(
        values=[(1, {varName + suf.termVariableSuffix: posDict[varName] for varName in posDict}) for posDict in
                entryPositionList],
        shape=[len(interpretationDict[variableName]) for variableName in variableNames],
        colors=[varName + suf.termVariableSuffix for varName in variableNames],
        name=name + suf.queryCoreSuffix
    ), startInterpretationDict


def query_result_to_entryPositionList(querySolution, interpretationDict=dict()):
    """
    For SPARQLWrapper in JSON Format
    InterpretationDict is further extended, when objects not appearing
    """
    projectedVariables = [str(variable) for variable in querySolution["head"]["vars"]]
    for variable in projectedVariables:
        if variable not in interpretationDict:
            interpretationDict[variable] = []

    entryPositionList = []
    for entry in querySolution["results"]["bindings"]:
        positionDict = {}
        for variable in projectedVariables:
            identifierString = str(entry[variable]["value"])
            if identifierString not in interpretationDict[variable]:
                interpretationDict[variable].append(identifierString)
            positionDict[variable] = interpretationDict[variable].index(identifierString)
        entryPositionList.append(positionDict)

    return entryPositionList, interpretationDict


def wrapper_json_evaluate_query(endpointString, queryString):
    sparql = SPARQLWrapper(endpointString)
    sparql.setQuery(queryString)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()
