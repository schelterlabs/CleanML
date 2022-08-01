#!/bin/bash
source venv-latest/bin/activate

python3 main.py --run_experiments --dataset USCensus --error_type missing_values #--run_analysis
python3 main.py --run_experiments --dataset USCensus --error_type outliers       #--run_analysis
# python3 main.py --run_experiments --dataset USCensus --error_type mislabel       #--run_analysis

python3 main.py --run_experiments --dataset ACSIncome --error_type missing_values #--run_analysis
python3 main.py --run_experiments --dataset ACSIncome --error_type outliers       #--run_analysis
# python3 main.py --run_experiments --dataset ACSIncome --error_type mislabel       #--run_analysis

python3 main.py --run_experiments --dataset Cardio --error_type outliers #--run_analysis
# python3 main.py --run_experiments --dataset Cardio --error_type mislabel #--run_analysis

python3 main.py --run_experiments --dataset Credit --error_type missing_values #--run_analysis
python3 main.py --run_experiments --dataset Credit --error_type outliers       #--run_analysis

# TODO
# python3 main.py --run_experiments --dataset GermanCredit --error_type ??? #--run_analysis

# python3 main.py --run_experiments --dataset StopFrisk --error_type missing_values #--run_analysis
# python3 main.py --run_experiments --dataset StopFrisk --error_type outliers       #--run_analysis
# python3 main.py --run_experiments --dataset StopFrisk --error_type mislabel       #--run_analysis

# python3 main.py --run_experiments --dataset Student --error_type ??? #--run_analysis
