import sys

from cleanlab.classification import CleanLearning
from numba import njit, prange
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

import utils


DATASETS = {
    "USCensus": {
        "categorical": ["Workclass", "Education", "Marital-status", "Occupation", "Relationship", "Race", "Sex", "Native-country"],
        "numerical": ["Age", "Hours-per-week", "Capital-gain", "Capital-loss"],
        "positive_class": ">50k",
    },
    "ACSIncome": {
        "categorical": ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"],
        "numerical": ["AGEP", "WKHP"],
        "positive_class": 1,
    },
    "Cardio": {
        "categorical": ["gender", "smoke", "alco", "active"],
        "numerical": ["age", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc"],
        "positive_class": 1,
    },
    "Credit": {
        "categorical": [],
        "numerical": ["RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"],
        "positive_class": 1,
    },
    "GermanCredit": {
        "categorical": ["status", "credit_history", "purpose", "savings", "employment", "other_debtors", "property", "installment_plans", "housing", "skill_level", "telephone", "foreign_worker", "personal_status"],
        "numerical": ["month", "credit_amount", "investment_as_income_percentage", "residence_since", "age", "number_of_credits", "people_liable_for"], 
        "positive_class": 1,
    },
}


def main():
    detection_method = sys.argv[1]

    for dataset in DATASETS:
        dataset_config = utils.get_dataset(dataset)
        dataset_filename = utils.get_dir(dataset_config, "raw", "raw.csv")
        df = pd.read_csv(dataset_filename)

        categorical_cols = DATASETS[dataset]["categorical"]
        numerical_cols = DATASETS[dataset]["numerical"]
        positive_class = DATASETS[dataset]["positive_class"]

        target = dataset_config["label"]
        labels = df[target]
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        if detection_method == "shapley":
            detected = detect_mislabeled_via_shapley(df, categorical_cols, numerical_cols, target, positive_class)
        elif detection_method == "cleanlab":
            detected = detect_mislabeled_via_cleanlab(df, categorical_cols, numerical_cols, encoded_labels)
        else:
            sys.exit("Please specify either 'shapley' or 'cleanlab'")

        print(f"[{dataset}] Detected {detected.sum()} of {df.shape[0]} mislabeled rows")

        incorrect_labels = df.iloc[detected][target]
        corrected_labels = label_encoder.inverse_transform(1 - label_encoder.transform(incorrect_labels))

        clean_df = df.copy()
        clean_df.loc[detected, target] = corrected_labels

        clean_filename = utils.get_dir(dataset_config, "raw", "mislabel_clean_raw.csv")
        clean_df.to_csv(clean_filename, index=False)


def detect_mislabeled_via_shapley(data, categorical_cols, numerical_cols, target, positive_class, seed=42):

    @njit(fastmath=True, parallel=True)
    def _compute_shapley_values(X_train, y_train, X_test, y_test, K):
        N = len(X_train)
        M = len(X_test)
        result = np.zeros(N, dtype=np.float32)

        for j in prange(M):
            score = np.zeros(N, dtype=np.float32)
            dist = np.zeros(N, dtype=np.float32)
            div_range = np.arange(1.0, N)
            div_min = np.minimum(div_range, K)
            for i in range(N):
                dist[i] = np.sqrt(np.sum(np.square(X_train[i] - X_test[j])))
            indices = np.argsort(dist)
            y_sorted = y_train[indices]
            eq_check = (y_sorted == y_test[j]) * 1.0
            diff = - 1 / K * (eq_check[1:] - eq_check[:-1])
            diff /= div_range
            diff *= div_min
            score[indices[:-1]] = diff
            score[indices[-1]] = eq_check[-1] / N
            score[indices] += np.sum(score[indices]) - np.cumsum(score[indices])
            result += score / M

        return result

    numerical_encoder = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaled_numeric", StandardScaler()),
    ])
    categorical_encoder = Pipeline([
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ("scale", StandardScaler()),
    ])

    encoder = ColumnTransformer(transformers=[
        ("categorical_features", categorical_encoder, categorical_cols),
        ("scaled_numeric", numerical_encoder, numerical_cols)], sparse_threshold=0.0)

    labels = np.array(data[target] == positive_class)

    # This is still a flaw, as we will ignore the 100 randomly chosen rows
    train_data, test_data = train_test_split(data, test_size=100, stratify=labels, random_state=seed)

    X_train = encoder.fit_transform(train_data)
    y_train = np.array(train_data[target] == positive_class)
    X_test = encoder.transform(test_data)
    y_test = np.array(test_data[target] == positive_class)

    shapley_values = _compute_shapley_values(X_train, np.squeeze(y_train), X_test, np.squeeze(y_test), K=10)

    return np.where(shapley_values < 0.0, 1, 0)


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
