from cleanlab.classification import CleanLearning
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

import utils


def main():
    datasets = {
        "USCensus": {
            "categorical": ["Workclass", "Education", "Marital-status", "Occupation", "Relationship", "Race", "Sex", "Native-country"],
            "numerical": ["Age", "Hours-per-week", "Capital-gain", "Capital-loss"],
        },
        "ACSIncome": {
            "categorical": ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"],
            "numerical": ["AGEP", "WKHP"],
        },
        "Cardio": {
            "categorical": ["gender", "smoke", "alco", "active"],
            "numerical": ["age", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc"],
        },
        "Credit": {
            "categorical": [],
            "numerical": ["RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"],
        },
        "GermanCredit": {
            "categorical": ["status", "credit_history", "purpose", "savings", "employment", "other_debtors", "property", "installment_plans", "housing", "skill_level", "telephone", "foreign_worker", "personal_status"],
            "numerical": ["month", "credit_amount", "investment_as_income_percentage", "residence_since", "age", "number_of_credits", "people_liable_for"], 
        },
    }
    for dataset in datasets:
        dataset_config = utils.get_dataset(dataset)
        dataset_filename = utils.get_dir(dataset_config, "raw", "raw.csv")
        df = pd.read_csv(dataset_filename)

        categorical_cols = datasets[dataset]["categorical"]
        numerical_cols = datasets[dataset]["numerical"]

        target = dataset_config["label"]
        labels = df[target]
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        detected = detect_mislabeled_via_cleanlab(df, categorical_cols, numerical_cols, encoded_labels)
        print(f"[{dataset}] Detected {detected.sum()} of {df.shape[0]} mislabeled rows")

        incorrect_labels = df.iloc[detected][target]
        corrected_labels = label_encoder.inverse_transform(1 - label_encoder.transform(incorrect_labels))

        clean_df = df.copy()
        clean_df.loc[detected, target] = corrected_labels

        clean_filename = utils.get_dir(dataset_config, "raw", "mislabel_clean_raw.csv")
        clean_df.to_csv(clean_filename, index=False)


def detect_mislabeled_via_cleanlab(data, categorical_cols, numerical_cols, labels):
    numerical_encoder = Pipeline([("imputer", SimpleImputer(strategy="mean")),
                                  ("scaled_numeric", StandardScaler())])

    encoder = ColumnTransformer(transformers=[
        ("categorical_features", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols),
        ("numerical_features", numerical_encoder, numerical_cols)])

    X = encoder.fit_transform(data)
    y = labels

    model = SGDClassifier(loss="log_loss").fit(X, y)
    cl = CleanLearning(model)

    label_issues = cl.find_label_issues(X, y)
    label_issue_indexes = np.array(label_issues.is_label_issue)

    return label_issue_indexes


if __name__ == "__main__":
    main()
