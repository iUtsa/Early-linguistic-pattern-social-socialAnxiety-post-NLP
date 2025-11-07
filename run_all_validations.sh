#!/bin/bash

echo "================================================================================"
echo "RUNNING ALL VALIDATION AND FIX SCRIPTS"
echo "================================================================================"
echo ""

# Activate virtual environment if needed
# source venv/bin/activate

echo "STEP 1: Running comprehensive validation..."
echo "--------------------------------------------------------------------------------"
python3 validate_and_fix.py
echo ""
read -p "Press Enter to continue to Step 2..."
echo ""

echo "STEP 2: Checking DAIC-WOZ for GAD-7..."
echo "--------------------------------------------------------------------------------"
python3 check_daic_woz_gad7.py
echo ""
read -p "Press Enter to continue to Step 3..."
echo ""

echo "STEP 3: Running keyword masking ablation..."
echo "--------------------------------------------------------------------------------"
echo "This may take 10-20 minutes..."
python3 keyword_masking_ablation.py
echo ""
read -p "Press Enter to continue to Step 4..."
echo ""

echo "STEP 4: Running balanced early-slice recomputation..."
echo "--------------------------------------------------------------------------------"
echo "This may take 20-30 minutes..."
python3 balanced_early_slice.py
echo ""

echo "================================================================================"
echo "ALL VALIDATIONS COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved in:"
echo "  - results/keyword_masking_ablation.csv"
echo "  - results/balanced_early_slice.csv"
echo ""
echo "Next steps:"
echo "  1. Review validation outputs above"
echo "  2. Check results CSV files"
echo "  3. Update paper based on findings"
echo ""

