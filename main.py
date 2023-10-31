import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


def loading_data():
    # x train
    custom_column_names = ["instance", "feature", "value"]
    df_train_x = pd.read_csv("train_X.csv", names=custom_column_names)
    df_pivot = df_train_x.pivot(index="instance", columns="feature", values="value").fillna(0)

    # Reset the index to make "instance" a regular column
    df_pivot.reset_index(inplace=True)

    # Rename the columns for clarity (0 and 1 instead of 0.0 and 1.0)
    df_pivot.columns = [f"feature_{col}" if col != "instance" else col for col in df_pivot.columns]

    # Optional: Convert "instance" column to an integer (if it's not already)
    df_pivot["instance"] = df_pivot["instance"]
    # y train
    custom_Y_names = ["poisonous"]
    df_train_Y = pd.read_csv("train_Y.csv", names=custom_Y_names)

    # x test
    df_test_x = pd.read_csv("test_X.csv", names=custom_column_names)
    df_pivot_test = df_test_x.pivot(index="instance", columns="feature", values="value").fillna(0)

    # Reset the index to make "instance" a regular column
    df_pivot_test.reset_index(inplace=True)

    # Rename the columns for clarity (0 and 1 instead of 0.0 and 1.0)
    df_pivot_test.columns = [f"feature_{col}" if col != "instance" else col for col in df_pivot_test.columns]

    # Optional: Convert "instance" column to an integer (if it's not already)
    df_pivot_test["instance"] = df_pivot_test["instance"]

    # y test

    df_test_Y = pd.read_csv("test_Y.csv", names=custom_Y_names)

    # drop instance column
    df_pivot.drop('instance', axis=1, inplace=True)
    df_pivot_test.drop('instance', axis=1, inplace=True)

    return df_pivot, df_train_Y, df_pivot_test, df_test_Y


def SVM_l2_regularization_Kfold(x_train, y_train):
    class_weights = {-1: 1, 1: 5}

    k = 5  # Number of folds for cross-validation
    C_values = [0.001, 0.01, 0.1, 1, 10]
    mean_scores = []

    for c in C_values:
        svm_classifier = SVC(kernel='linear', C=c, class_weight=class_weights)
        kf = KFold(n_splits=k)
        scores = cross_val_score(svm_classifier, x_train, y_train, cv=kf)
        mean_accuracy = np.mean(scores)
        mean_scores.append(mean_accuracy)
        print(f'C={c}: Mean Accuracy = {mean_accuracy:.2f}')
    plt.figure(figsize=(8, 6))
    plt.plot(C_values, mean_scores, marker='o', linestyle='-')
    plt.title("Mean Accuracy vs. C Values")
    plt.xlabel("C Values")
    plt.ylabel("Mean Accuracy")
    plt.xscale('log')  # Use a log scale for the x-axis if C values vary widely
    plt.grid(True)
    plt.show()
    return mean_scores


def SVM_Prediction(x_train, y_train, x_test, y_test):
    class_weights = {-1: 1, 1: 5}
    svm_classifier = SVC(kernel='linear', C=0.01, class_weight=class_weights)
    svm_classifier.fit(x_train, y_train)

    # Make predictions on the test data
    y_pred = svm_classifier.predict(x_test)

    # Calculate and print the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    plt.figure(figsize=(8, 6))

    feature_names = x_train.columns

    coefficients = svm_classifier.coef_

    feature_coefficients = dict(zip(feature_names, coefficients[0]))  # Use coefficients[0] for class -1

    #print(feature_coefficients)
    return feature_coefficients


def Lasso_Regularization(x_train, y_train, x_test, y_test):
    alpha = 0.01
    lasso = Lasso(alpha=alpha)
    lasso.fit(x_train, y_train)
    selected_features = [feature for feature, coef in enumerate(lasso.coef_) if coef != 0]
    #print(f'Selected features: {selected_features}')
    X_selected = x_train[:, selected_features]
    return X_selected


def Tuning_Lasso(x_train, y_train):
    alphas = np.logspace(-4, 4, 9)  # Varying from 0.0001 to 10,000

    # Create an empty list to store cross-validated scores
    cv_scores = []

    # Perform cross-validation for each alpha
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        scores = cross_val_score(lasso, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_scores.append(-scores.mean())  # Convert to positive MSE values

    # Find the alpha with the lowest MSE
    best_alpha = alphas[cv_scores.index(min(cv_scores))]
    best_mse = min(cv_scores)

    print(f"Best alpha: {best_alpha}")
    print(f"Lowest MSE: {best_mse}")
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, cv_scores, marker='o')
    plt.title('Cross-Validated Mean Squared Error vs. Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    plt.show()
    return best_alpha


def BoLasso(x_train, y_train, alpha=0.001):
    n_bootstraps = 1000
    selected_features = []

    for _ in range(n_bootstraps):
        # Generate a bootstrap sample
        X_boot, y_boot = resample(x_train, y_train, random_state=np.random.randint(0, 100))

        # Fit a Lasso regression model
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_boot, y_boot)
        selected_indices = np.where(lasso.coef_ != 0)[0]
        selected_features.append(selected_indices)

    feature_selection_count = np.bincount(np.concatenate(selected_features))

    threshold = n_bootstraps // 2
    final_selected_features = np.where(feature_selection_count >= threshold)[0]

    #print("Selected Features:", final_selected_features)
    X_selected = x_train.iloc[:, final_selected_features]
    return final_selected_features, X_selected


def Lasso_SVM(x_train, y_train, x_test, y_test):
    Lasso_selected_features, X_Lasso_selected_features = BoLasso(x_train, y_train)
    extra_columns_in_test = [col for col in x_train.columns if col not in X_Lasso_selected_features]

    # Drop these extra columns from the training dataset
    x_train.drop(extra_columns_in_test, axis=1, inplace=True)
    x_test.drop(extra_columns_in_test, axis=1, inplace=True)
    feature_coefficients = SVM_Prediction(x_train, y_train, x_test, y_test)
    return feature_coefficients

def scatter_plot_Coefficients(x_train, y_train, x_test, y_test):
    feature_coefficients_SVM = SVM_Prediction(x_train, y_train, x_test, y_test)

    # Tuning_Lasso(x_train, y_train)
    feature_coefficients_Lasso = Lasso_SVM(x_train, y_train, x_test, y_test)
    for key in feature_coefficients_SVM:
        if key not in feature_coefficients_Lasso:
            feature_coefficients_Lasso[key] = 0
    sorted_feature_coefficients_Lasso = {key: feature_coefficients_Lasso[key] for key in feature_coefficients_SVM.keys()}
    print(feature_coefficients_SVM)
    print(sorted_feature_coefficients_Lasso)
    x_values = list(feature_coefficients_SVM.values())
    y_values = list(sorted_feature_coefficients_Lasso.values())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.scatter(x_values, y_values, marker='o', c='r', label='Coefficients')

    # Set axis labels and a title
    plt.xlabel('Coefficients of model 1')
    plt.ylabel('Coefficients of model 2')
    plt.title('Comparing Coefficients L2 and L1')
    plt.legend()
    plt.show()

def Random_Forest_tuning_prediction(x_train, y_train, x_test, y_test):
    random_forest = RandomForestClassifier(random_state=42)

    # hyperparameters to search
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 15, 25],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # grid search to find the best hyperparameters
    grid_search = GridSearchCV(random_forest, param_grid, cv=5)
    grid_search.fit(x_train, y_train)

    #best hyperparameters
    best_params = grid_search.best_params_
    # grid_results = grid_search.cv_results_
    # print("This is grid search", grid_results)
    best_random_forest = RandomForestClassifier(random_state=42, n_estimators = 50, max_depth= 5 ,min_samples_split = 2 ,min_samples_leaf = 1)
    best_random_forest.fit(x_train, y_train)
    y_pred = best_random_forest.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best Model Accuracy: {accuracy:.2f}")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)


def main():
    x_train, y_train, x_test, y_test = loading_data()
    y_train = y_train.values.ravel()
    # extra_columns_in_train = [col for col in x_train.columns if col not in x_test.columns]
    # for col in extra_columns_in_train:
    #     x_test[col] = 0
    # mean_scores = SVM_l2_regularization_Kfold(x_train, y_train)
    extra_columns_in_test = [col for col in x_train.columns if col not in x_test.columns]

    # Drop these extra columns from the training dataset
    x_train.drop(extra_columns_in_test, axis=1, inplace=True)
    #scatter_plot_Coefficients(x_train, y_train, x_test, y_test)
    Random_Forest_tuning_prediction(x_train, y_train, x_test, y_test)



if __name__ == "__main__":
    main()