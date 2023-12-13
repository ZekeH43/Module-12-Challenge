# Module-12-Challenge
## Overview of the Analysis

The purpose of this analysis is to address the challenge of credit risk classification, a problem inherently characterized by imbalanced classes. The goal is to build a machine learning model using logistic regression to predict the creditworthiness of borrowers. Given the nature of credit risk, where healthy loans significantly outnumber risky loans, the analysis focuses on techniques to handle imbalanced classes and evaluates the model's performance in predicting high-risk loans.

The dataset used contains historical lending activity from a peer-to-peer lending services company. The primary financial information includes features such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt. The target variable, "loan_status," is binary, where 0 denotes a healthy loan, and 1 indicates a high risk of default. The objective is to predict the likelihood of a loan being high-risk based on the given financial features.

1.Data Preprocessing: Reading the dataset from a CSV file, separating features and labels, and checking the balance of the target variable.

2.Data Splitting: Dividing the data into training and testing sets using the train_test_split function.

3.Model Training (Original Data): Fitting a logistic regression model using the original training data.

4.Model Evaluation (Original Data): Predicting on the testing data and evaluating the model's performance using accuracy, confusion matrix, and classification report metrics.

5.Resampling (Oversampling): Using RandomOverSampler from imbalanced-learn to address class imbalance.

6.Model Training (Resampled Data): Fitting a logistic regression model using the oversampled training data.

7.Model Evaluation (Resampled Data): Predicting on the testing data and evaluating the performance metrics for the resampled model.

Logistic Regression: Utilized logistic regression as the primary machine learning model due to its effectiveness in binary classification problems.
RandomOverSampler: Applied oversampling using the RandomOverSampler module from imbalanced-learn to address the imbalance in the target variable. This technique ensures that both classes have an equal number of data points, providing a more balanced training set for the model.
## Results

Machine Learning Model 1:

-Balanced Accuracy Score: 94.4%

-Precision (High-Risk Loans): 87%

-Recall (High-Risk Loans): 89%

-Precision (Healthy Loans): 100%

-Recall (Healthy Loans): 100%


Machine Learning Model 2:

-Balanced Accuracy Score: 99.6%

-Precision (High-Risk Loans): 87%

-Recall (High-Risk Loans): 100%

-Precision (Healthy Loans): 100%

-Recall (Healthy Loans): 100%

## Summary

The machine learning models were evaluated based on their balanced accuracy, precision, and recall scores for predicting credit risk. The two models, trained on the original and oversampled data, exhibited distinct performances.

Model 1: Achieved a balanced accuracy of 94.4%, demonstrating strong predictive capabilities. It maintained a good balance between precision and recall for both high-risk and healthy loans.

Model 2: Outperformed Model 1 with a remarkable balanced accuracy of 99.6%. It significantly improved precision and recall for high-risk loans, achieving near-perfection in identifying high-risk loans without introducing false positives.

Recommendation:

Considering the exceptional performance of Model 2, especially in correctly identifying high-risk loans, it is recommended for use in credit risk classification. The oversampling technique employed in Model 2, using RandomOverSampler, effectively addressed the imbalance in the dataset and led to a more robust model.
