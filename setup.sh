#!/bin/bash

# Create and activate a virtual environment using python venv
python3 -m venv venv-phase-1
source venv-phase-1/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the required packages and dependencies
pip install -r "$(dirname "$0")/requirements.txt"

echo ""
echo ""
echo "Next steps: "
echo "1. Copy the raw data in '$(dirname "$0")/data/raw/'!"
echo "   The data can be downloaded from https://www.drivendata.org/competitions/57/nepal-earthquake/"
echo ""
echo "2. Activate the environment with: "
echo "   source venv-phase-1/bin/activate"
echo ""
echo "3. Run the prediction pipeline with default settings:"
echo "   python3 $(dirname "$0")/src/main.py --config configs/config.yml"
echo ""
echo "Optional: Adapt the configuration file (configs/config.yml) to your specific needs and run the pipeline with it. "
echo ""
echo ""
