# Anomaly Detection

A machine learning project that detects anomalies in structured datasets using a Decision Tree classifier, achieving ~75% accuracy on the validation set.

## Overview

This project explores anomaly detection through supervised learning. The goal was to build a model capable of identifying irregular patterns in structured data — distinguishing between normal and anomalous samples.

## Approach

- Split the training data into train and validation sets (80/20)
- Trained a Decision Tree classifier with tuned hyperparameters
- Evaluated model performance using accuracy score on the validation set
- Generated predictions on unseen test data

## Results

|Metric               | Value                           |
|---------------------|---------------------------------|
| Validation Accuracy | ~75%                            |
| Model               | Decision Tree Classifier        |
| Best Parameters     | max_depth=10, criterion=entropy |

## Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy

## Note

The dataset is not included in this repository as it was provided as part of a university assignment.
