import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=20)

# Average CV score on the training set was: 0.8647474747474747
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=MultinomialNB(alpha=10.0, fit_prior=True)),
    RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.45, min_samples_leaf=19, min_samples_split=7, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 20)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)