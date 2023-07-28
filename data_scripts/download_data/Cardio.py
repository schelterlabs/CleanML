import os

import pandas as pd


os.makedirs("data/Cardio/raw", exist_ok=True)

# Download from Kaggle:
# https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
data = pd.read_csv("cardio_train.csv", sep=";", index_col="id")

data.to_csv("data/Cardio/raw/raw.csv", index=False)
