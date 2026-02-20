import tarfile, random
from pathlib import Path

"""
Here we load instances of 3-SAT in the DIMACS format from https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html.
"""
def list_cnf_members(tar_path: Path):
    """Return tar members that are .cnf files."""
    with tarfile.open(tar_path, "r") as tar:
        return [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".cnf")]


def read_3sat_instances(tar_path: Path, instance_index: int):
    """Extract and print the instance_index-th 3-SAT instance from .cnf files."""
    with tarfile.open(tar_path, "r") as tar:
        members = list_cnf_members(tar_path)
        if instance_index < 0 or instance_index >= len(members):
            raise IndexError("Instance index out of range.")
        member = members[instance_index]
        with tar.extractfile(member) as file:
            lines = file.readlines()
            clauses = []
            for line in lines:
                line = line.decode("utf-8").strip()
                if line.startswith("p cnf"):
                    _, _, num_variables, num_clauses = line.split()
                elif not line.startswith("c") and not line.startswith("%") and line:
                    clause = list(map(int, line.split()))
                    if clause[-1] == 0:  # Remove the terminating 0 in DIMACS format
                        clause.pop()
                    clauses.append(clause)
    return clauses


def dimacs_to_clauseList(dimacsClauses):
    return [{str(abs(lit)): int(lit > 0) for lit in clause} for clause in dimacsClauses]


if __name__ == "__main__":
    """
    Example of a conversion of a 3-SAT instance in DIMACS format into a Computation-Activation Network
    """

    from demonstrations.sat import sat_constraints as satc

    DATASET_PATH = Path("./uf20-91.tar")
    fileNumber = 10

    caNetwork = satc.get_clauseList_as_CANetwork(dimacs_to_clauseList(read_3sat_instances(DATASET_PATH, fileNumber)))
