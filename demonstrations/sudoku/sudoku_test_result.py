from demonstrations.sudoku import sudoku_constraints as scons
from tnreason import engine

def check_solution_by_contraction(result, num=3, tol=1e-12):
    # Build CANetwork with result as evidence
    ca = scons.get_assignment_as_CANetwork(num=num, startAssignment=result)

    # Contract full tensor network (constraints + activations)
    z = engine.contract(ca.create_cores(), openColors=[])[:]
    # alternatively: z = ca.get_partition_function()

    is_valid = z > tol
    return is_valid, z
