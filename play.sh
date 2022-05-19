#!/bin/bash
source venv/bin/activate

python3 main.py --run_experiments --dataset USCensus --error_type missing_values --run_analysis
