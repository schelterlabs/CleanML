import os

import pandas as pd


os.makedirs("data/GermanCredit/raw", exist_ok=True)

# Download from UCI machine learning repository:
# https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
columns = ['status', 'month', 'credit_history', 'purpose', 'credit_amount', 'savings',
           'employment', 'investment_as_income_percentage', 'personal_status',
           'other_debtors', 'residence_since', 'property', 'age',
           'installment_plans', 'housing', 'number_of_credits',
           'skill_level', 'people_liable_for', 'telephone',
           'foreign_worker', 'credit']
data = pd.read_csv("german.data", sep=" ", header=None, names=columns, na_values=['A65', 'A124'])

data.to_csv("data/GermanCredit/raw/raw.csv", index=False)
