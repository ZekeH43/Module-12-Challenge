# Credit risk: catching more high-risk loans

This Python project asks what happens when a logistic regression model sees more examples of a rare class during training. It compares two models on the same held-out loans from the supplied course dataset. The only change in training strategy is whether to duplicate minority-class examples.

## Run the comparison

From the repository root, use Python 3.11 or 3.12:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python credit_risk.py
python -m unittest -v
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. The script reads the included CSV and prints the comparison; no account or dataset download is needed. To explore interactively, install JupyterLab with `python -m pip install jupyterlab` in the same environment, then run `jupyter lab credit_risk_resampling.ipynb` from the repository root.

## What the results show

| Training data | Balanced accuracy | High-risk precision | High-risk recall | Healthy loans flagged | High-risk loans missed |
|---|---:|---:|---:|---:|---:|
| Original | 98.64% | 87.16% | 97.76% | 90 | 14 |
| Oversampled | 99.58% | 86.53% | 99.68% | 97 | 2 |

On this split, oversampling caught 12 more high-risk loans while incorrectly flagging 7 more healthy loans. Recall rose, but precision fell slightly: finding more risky cases came at a cost. Balanced accuracy averages recall across both classes so that the more common healthy class does not dominate the score.

## Follow the code

1. [The dataset](Starter_Code/Resources/lending_data.csv) contains seven numeric inputs, including loan size, borrower income and total debt. The target is loan_status: 0 for healthy or 1 for high risk.
2. In [credit_risk.py](credit_risk.py), `compare` reserves a stratified 25% test set with a fixed seed. `balance_training` duplicates randomly selected rows from the minority class until the training classes are equally frequent.
3. Each model fits its own standard scaler and logistic regression using training data only. Both predict on the untouched test set. The [notebook](credit_risk_resampling.ipynb) calls that same code and shows the class counts alongside the results.
4. [The tests](test_credit_risk.py) check which examples are duplicated and verify the direction of false positives and missed cases using a hand-calculated example.

This is a classroom experiment on one random split. The CSV has no borrower identifier or time column to test performance on new borrowers or later loans. These results do not establish suitability for lending decisions.
