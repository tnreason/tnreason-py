from tnreason import representation
import numpy as np

def get_thermometer_constraint(thermo_cells, thermo_name="thermo"):
    """
    Creates a thermometer constraint for tnreason framework.
    
    Args:
        thermo_cells: List of (row, col) tuples defining thermometer path from bulb to tip
        thermo_name: Name identifier for this constraint
    
    Returns:
        Dictionary with thermometer constraint in tnreason format
    """
    if len(thermo_cells) < 2:
        raise ValueError("Thermometer must have at least 2 cells")
    
    # Generate all possible digit assignments for thermometer cells
    # For cells c₁, c₂, ..., cₙ: c₁ < c₂ < ... < cₙ
    valid_assignments = []
    
    def generate_thermo_sequences(positions, remaining_digits=range(1, 10)):
        if not positions:
            return [[]]
        
        first_pos = positions[0]
        sequences = []
        
        if len(positions) == 1:
            # Last cell - any remaining digit works
            for digit in remaining_digits:
                sequences.append([digit])
        else:
            # Not last cell - must leave room for increasing sequence
            for i, digit in enumerate(remaining_digits[:-1]):
                # Need at least len(positions)-1 digits greater than current
                if len(remaining_digits) - i - 1 >= len(positions) - 1:
                    for subsequence in generate_thermo_sequences(positions[1:], remaining_digits[i+1:]):
                        sequences.append([digit] + subsequence)
        
        return sequences
    
    # Generate all valid strictly increasing sequences
    valid_sequences = generate_thermo_sequences(thermo_cells)
    
    # Convert to tnreason constraint format
    variable_names = []
    for i, (row, col) in enumerate(thermo_cells):
        variable_names.append(f"r{row}c{col}")
    
    # Create constraint dictionary
    constraint = {
        thermo_name: {
            "type": "categorical",
            "variables": variable_names,
            "valid_assignments": valid_sequences
        }
    }
    
    return constraint

def add_thermometer_constraints(can_network, thermometer_list):
    """
    Adds multiple thermometer constraints to a ComputationActivationNetwork.
    
    Args:
        can_network: Existing ComputationActivationNetwork
        thermometer_list: List of dictionaries, each with:
            - 'cells': List of (row, col) tuples
            - 'name': Constraint name (optional)
    
    Returns:
        Updated ComputationActivationNetwork with thermometer constraints
    """
    new_constraints = {}
    
    for i, thermo_info in enumerate(thermometer_list):
        thermo_cells = thermo_info['cells']
        thermo_name = thermo_info.get('name', f'thermo_{i}')
        
        # Get thermometer constraint
        thermo_constraint = get_thermometer_constraint(thermo_cells, thermo_name)
        new_constraints.update(thermo_constraint)
    
    # Create categorical cores for new constraints
    from tnreason.application.categoricals_to_cores import create_categorical_cores
    new_cores = create_categorical_cores(new_constraints)
    
    # Add to existing network
    can_network.computationCoreDict.update(new_cores)
    
    return can_network

def get_binary_thermometer_constraints(thermo_cells, thermo_name="thermo"):
    """
    Creates thermometer constraint using binary inequalities between adjacent cells.
    This is more efficient for long thermometers.
    
    Args:
        thermo_cells: List of (row, col) tuples defining thermometer path
        thermo_name: Name identifier for this constraint
    
    Returns:
        Dictionary with binary inequality constraints
    """
    if len(thermo_cells) < 2:
        return {}
    
    constraints = {}
    
    # Create binary inequality constraints between adjacent cells
    for i in range(len(thermo_cells) - 1):
        (r1, c1) = thermo_cells[i]
        (r2, c2) = thermo_cells[i + 1]
        
        # Constraint: cell_i < cell_{i+1}
        constraint_name = f"{thermo_name}_inequality_{i}"
        
        # Generate all valid pairs where first < second
        valid_pairs = []
        for d1 in range(1, 10):
            for d2 in range(d1 + 1, 10):  # d2 must be > d1
                valid_pairs.append([d1, d2])
        
        constraints[constraint_name] = {
            "type": "categorical",
            "variables": [f"r{r1}c{c1}", f"r{r2}c{c2}"],
            "valid_assignments": valid_pairs
        }
    
    return constraints

def get_sudoku_thermometer_connection_constraints(num=3):
    """
    Creates constraints that connect boolean sudoku variables with categorical thermometer variables.
    
    The sudoku variables are: a_{row_box}_{row}_{col_box}_{col}_{number} (boolean)
    The thermometer variables are: r{row}c{col} (categorical, 1-9)
    
    Returns:
        Dictionary of connection constraints
    """
    connection_constraints = {}
    
    for row_box in range(num):
        for row in range(num):
            for col_box in range(num):
                for col in range(num):
                    # For each position, create a constraint connecting boolean and categorical variables
                    constraint_name = f"connection_r{row_box}r{row}c{col_box}c{col}"
                    
                    # Variables: all boolean variables for this position + the categorical variable
                    boolean_vars = [f"a_{row_box}_{row}_{col_box}_{col}_{n}" for n in range(1, num*num + 1)]
                    categorical_var = f"r{row_box * num + row}c{col_box * num + col}"
                    
                    # Valid assignments: exactly one boolean is true and categorical matches that number
                    valid_assignments = []
                    for n in range(1, num*num + 1):
                        # Create assignment where boolean for number n is true, others are false
                        # and categorical variable equals n
                        assignment = [0] * len(boolean_vars) + [n]
                        assignment[n-1] = 1  # Set the nth boolean to true
                        valid_assignments.append(assignment)
                    
                    connection_constraints[constraint_name] = {
                        "type": "categorical",
                        "variables": boolean_vars + [categorical_var],
                        "valid_assignments": valid_assignments
                    }
    
    return connection_constraints

def add_thermometer_to_sudoku(num=3, thermometers=None, evidence_dict=None):
    """
    Creates a complete Sudoku network with thermometer constraints.
    
    Args:
        num: Grid size (default 9 for 9x9 Sudoku)
        thermometers: List of thermometer definitions, each with 'cells' and optional 'name'
        evidence_dict: Initial evidence dictionary
    
    Returns:
        ComputationActivationNetwork with Sudoku + thermometer constraints
    """
    from . import sudoku_constraints as rep
    
    # Get base Sudoku constraints
    base_constraints = rep.get_sudoku_constraints(num)
    
    # Add thermometer constraints
    if thermometers is None:
        thermometers = []
    
    thermo_constraints = {}
    for i, thermo in enumerate(thermometers):
        thermo_cells = thermo['cells']
        thermo_name = thermo.get('name', f'thermo_{i}')
        
        # Use binary version for efficiency
        binary_constraints = get_binary_thermometer_constraints(thermo_cells, thermo_name)
        thermo_constraints.update(binary_constraints)
    
    # Add connection constraints between boolean and categorical variables
    connection_constraints = get_sudoku_thermometer_connection_constraints(num)
    
    # Combine all constraints
    all_constraints = {**base_constraints, **thermo_constraints, **connection_constraints}
    
    # Create categorical cores
    from tnreason.application.categoricals_to_cores import create_categorical_cores
    com_core_dict = create_categorical_cores(all_constraints)

    print("com_core_dict keys:", com_core_dict.keys())
    
    # Create evidence cores
    if evidence_dict is None:
        evidence_dict = {}
    
    can_param_dict = {
        evidence_key: representation.create_basis_core(
            name=evidence_key, 
            shape=[2], 
            colors=[evidence_key], 
            numberTuple=[evidence_dict[evidence_key]]
        )
        for evidence_key in evidence_dict
    }
    
    # Create feature dictionary
    try:
        feature_dict = representation.standard_featureDict_from_computationCoreDict(com_core_dict)
    except:
        from tnreason.representation.ca_network import standard_featureDict_from_computationCoreDict
        feature_dict = standard_featureDict_from_computationCoreDict(com_core_dict)
    
    # Create network
    ca_network = representation.ComputationActivationNetwork(
        featureDict=feature_dict,
        computationCoreDict=com_core_dict,
        canParamDict=can_param_dict
    )
    
    return ca_network

# Example usage for Tendril Battle puzzle
def create_tendril_battle_network(evidence_dict=None):
    """
    Creates the network for the Tendril Battle puzzle.
    """
    # Define thermometer cells from the visual elements
    thermometers = [
        {
            'name': 'thermo_r7c1_path1',
            'cells': [(6, 0), (5, 0), (4, 0), (3, 0)]  # r7c1, r6c1, r5c1, r4c1
        },
        {
            'name': 'thermo_r7c1_path2', 
            'cells': [(6, 0), (5, 1), (4, 2), (4, 1), (2, 2)]  # r7c1, r6c2, r5c3, r4c2, r3c3
        },
        {
            'name': 'thermo_r7c1_path3',
            'cells': [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4)]  # r7c1, r7c2, r7c3, r7c4, r7c5
        },
        {
            'name': 'thermo_r7c1_path4',
            'cells': [(6, 0), (7, 0), (7, 1), (8, 2), (8, 3)]  # r7c1, r8c1, r8c2, r9c3, r9c4
        },
        {
            'name': 'thermo_r2c7_path1',
            'cells': [(1, 6), (2, 7), (3, 7), (4, 7), (5, 8)]  # r2c7, r3c8, r4c8, r5c9, r6c9
        },
        {
            'name': 'thermo_r2c7_path2',
            'cells': [(1, 6), (2, 5), (2, 4)]  # r2c7, r3c6, r3c5
        },
        {
            'name': 'thermo_r2c7_path3',
            'cells': [(1, 6), (1, 5), (0, 4), (1, 3), (1, 2), (1, 1)]  # r2c7, r2c6, r1c5, r2c4, r2c3, r2c2
        },
        {
            'name': 'thermo_r2c7_path4',
            'cells': [(1, 6), (2, 6), (3, 6), (4, 5), (5, 5)]  # r2c7, r3c7, r4c7, r5c6, r6c6
        }
    ]
    
    return add_thermometer_to_sudoku(num=3, thermometers=thermometers, evidence_dict=evidence_dict)

if __name__ == "__main__":
    # Test with a simple 3x3 example
    test_thermo = {
        'name': 'test_thermo',
        'cells': [(0, 0), (0, 1), (0, 2)]  # Simple 3-cell thermometer
    }
    
    # Create network
    network = add_thermometer_to_sudoku(num=3, thermometers=[test_thermo])
    
    print("Thermometer constraint network created successfully!")
    print(f"Number of constraints: {len(network.computationCoreDict)}")