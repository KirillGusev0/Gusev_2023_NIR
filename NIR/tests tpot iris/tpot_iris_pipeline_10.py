import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=63)

# Average CV score on the training set was: 0.990909090909091
exported_pipeline = make_pipeline(
    Nystroem(gamma=0.7000000000000001, kernel="additive_chi2", n_components=1),
    DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=6, min_samples_split=19)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 63)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
