from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

# Load the Iris dataset (or replace with your dataset loading)
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1)  # You can adjust kernel and C as needed

# Define the number of folds
num_folds = 5

# Initialize a KFold object for k-fold cross-validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

accuracy_scores = []

# Perform k-fold cross-validation
for train_index, test_index in kf.split(X):
    print(train_index,test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the SVM classifier on the training data
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = svm_classifier.predict(X_test)

    # Calculate accuracy for this fold
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Calculate the average accuracy over all folds
average_accuracy = np.mean(accuracy_scores)

# Print the average accuracy
print(f'Average Accuracy: {average_accuracy:.2f}')
