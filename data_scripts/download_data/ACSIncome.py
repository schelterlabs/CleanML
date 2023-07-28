"""
Download folktables data for the state of California in the year 2018.
Keep only the columns used for the ACSIncome binary classification task.

python3 -m venv folktables-env
source folktables-env/bin/activate
pip install -U pip folktables
python3 [this script]  # from project root
"""

import os

import numpy as np

from folktables import ACSDataSource


os.makedirs("data/ACSIncome/raw", exist_ok=True)

data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
acs_data = data_source.get_data(states=["CA"], download=True)

acs_data["label"] = np.where(acs_data["PINCP"] > 50000, 1, 0)

columns = ["label", "AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]
data = acs_data[columns]

data.to_csv("data/ACSIncome/raw/raw.csv", index=False)
