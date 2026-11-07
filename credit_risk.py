"""Compare two logistic regression models on the same held-out loans."""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "Starter_Code" / "Resources" / "lending_data.csv"
TARGET = "loan_status"
SEED = 1


def balance_training(rows):
    """Duplicate minority examples until both classes have the same count."""
    groups = [group for _, group in rows.groupby(TARGET)]
    largest = max(map(len, groups))
    extras = [g.sample(largest - len(g), replace=True, random_state=SEED) for g in groups]
    return pd.concat([rows, *extras])


def score(actual, predicted):
    """Class 1 is high risk; a false positive is a healthy loan flagged."""
    _, fp, fn, _ = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "balanced accuracy": balanced_accuracy_score(actual, predicted),
        "precision (1)": precision_score(actual, predicted, zero_division=0),
        "recall (1)": recall_score(actual, predicted, zero_division=0),
        "false positives": fp,
        "missed high-risk": fn,
    }


def compare(rows):
    """Split once; fit scaling and each classifier only on its training rows."""
    train, test = train_test_split(rows, test_size=0.25, stratify=rows[TARGET], random_state=SEED)
    scores = {}
    for name, sample in [("Original", train), ("Oversampled", balance_training(train))]:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(sample.drop(columns=TARGET), sample[TARGET])
        predictions = model.predict(test.drop(columns=TARGET))
        scores[name] = score(test[TARGET], predictions)
    return pd.DataFrame.from_dict(scores, orient="index")


if __name__ == "__main__":
    print(compare(pd.read_csv(DATA)).round(4).to_string())
