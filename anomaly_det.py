import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# load training data
with np.load('train.npz') as data:
    x_train, y_train = data['x_train'], data['y_train']

# load test data
with np.load('test.npz') as data:
    x_test = data['x_test']

# split training data into train and validation sets
x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# train Decision Tree model with optimized hyperparameters
model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=3, criterion='entropy', random_state=0)
model.fit(x_train_split, y_train_split)

# evaluate model on validation set
y_val_pred = model.predict(x_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation accuracy: {accuracy}")

# generate predictions on test set
predictions = model.predict(x_test)

# export predictions to CSV
submission = pd.DataFrame({
    'Id': np.arange(len(predictions)),
    'Label': predictions
})
submission.to_csv('predictions.csv', index=False)
print("Predictions file created successfully!")
