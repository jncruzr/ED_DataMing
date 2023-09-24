import pandas as pd
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import cohen_kappa_score, roc_auc_score, f1_score

# Load the dataset (replace 'ca1-dataset.csv' with the actual path)
data = pd.read_csv('ca1-dataset.csv')

# Preprocess the data
data['OffTask'] = data['OffTask'].map({'N': 0, 'Y': 1})

# Split data into features (X) and target variable (y)
X = data.drop(['OffTask','Unique-id', 'namea', 'Avghelp', 'Avgchoice', 'Avgstring', 'Avgnumber', 'Avgpoint', 'Avghelppct-up', 'Avgrecent8help', 'AvgasymptoteA-up', 'AvgasymptoteB-up'], axis=1)
y = data['OffTask']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = [
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Gaussian Naive Bayes', GaussianNB()),
    ('XGBoost', XGBClassifier(random_state=42))
]

best_kappa = -1.0  # Initialize best Kappa score
best_features = None  # Initialize the list of best features

# Generate all combinations of columns
all_columns = list(X.columns)
for r in range(1, len(all_columns) + 1):
    column_combinations = combinations(all_columns, r)
    for columns in column_combinations:
        # Subset the training data with selected columns
        X_train_subset = X_train[list(columns)]

        # Iterate through classifiers
        for name, clf in classifiers:
            if name == 'XGBoost':
                # Define hyperparameter grid for XGBoost
                param_grid = {
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.1, 0.01, 0.001],
                    'n_estimators': [100, 200, 300]
                }

                # Perform GridSearchCV for hyperparameter tuning
                grid_search = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
                grid_search.fit(X_train_subset, y_train)

                # Get the best estimator with optimal hyperparameters
                clf = grid_search.best_estimator_

            clf.fit(X_train_subset, y_train)
            y_pred = clf.predict(X_train_subset)

            # Calculate Kappa
            kappa = cohen_kappa_score(y_train, y_pred)

            # If Kappa is better, update best Kappa and features
            if kappa > best_kappa:
                best_kappa = kappa
                best_features = list(columns)

# Print the list of variables that produced the largest Kappa and the Kappa value
print("Best Kappa:", best_kappa)
print("Best Features:", best_features)

# Use the best features to cross-validate on the test dataset
X_test_subset = X_test[best_features]

# Iterate through classifiers and use the best features
for name, clf in classifiers:
    clf.fit(X_train_subset, y_train)
    y_pred = clf.predict(X_test_subset)

    # Calculate Kappa on the test dataset
    kappa = cohen_kappa_score(y_test, y_pred)

    # Print the evaluation metrics for each classifier
    print(f"Classifier: {name}")
    print("Kappa:", kappa)
    print("-" * 30)
