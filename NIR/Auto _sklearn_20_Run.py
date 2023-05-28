# -*- coding: utf-8 -*-
"""
Created on Sun May 28 16:00:58 2023

@author: kirdr
"""

from pprint import pprint

import sklearn.datasets
import sklearn.metrics

import autosklearn.classification


############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=42
)

############################################################################
# Build and fit a classifier
# ==========================

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=10,
    per_run_time_limit=10,
    tmp_folder="/tmp/autosklearn_classification_example_tmp",
)
automl.fit(X_train, y_train, dataset_name="iris")

############################################################################
# View the models found by auto-sklearn
# =====================================

print(automl.leaderboard())

############################################################################
# Print the final ensemble constructed by auto-sklearn
# ====================================================

pprint(automl.show_models(), indent=4)

###########################################################################
# Get the Score of the final ensemble
# ===================================

predictions = automl.predict(X_test)
print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))