#!/bin/bash
# Run all Sudoku constraint validation tests
# Usage: ./test_constraints.sh

echo "============================================================"
echo "Sudoku Variant Constraints - Test Suite"
echo "============================================================"
echo ""

cd "$(dirname "$0")"

# Run main validation suite
python validation/test_all_constraints.py

echo ""
echo "============================================================"
echo "Individual Constraint Self-Tests"
echo "============================================================"

for file in constraints/*.py; do
    if [[ $(basename "$file") != "__init__.py" ]]; then
        echo -n "Testing $(basename "$file") ... "
        python "$file" > /dev/null 2>&1 && echo "✓" || echo "✗"
    fi
done

echo "============================================================"
