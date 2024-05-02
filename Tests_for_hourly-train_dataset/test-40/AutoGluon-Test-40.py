import matplotlib.pyplot as plt
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

df = pd.read_csv(r"C:\Users\kirdr\.spyder-py3\TimeDataSets\Hourly-train-processed.csv")
df.head()

train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)
train_data.head()


predictor = TimeSeriesPredictor(
    prediction_length=48,
    target="target",
    eval_metric="MSE",
)
 
predictor.fit(
    train_data,
    presets="high_quality",
    time_limit=3600,
    num_val_windows=5,
    enable_ensemble=False,
    #val_step_size = 2,
    hyperparameters={
      #"Theta": {},
      "TemporalFusionTransformer": {},
      #"DirectTabular": {},
      #"RecursiveTabular": {},
   },
)

predictions = predictor.predict(train_data)
predictions.head()

# TimeSeriesDataFrame can also be loaded directly from a file
test_data = TimeSeriesDataFrame.from_path(r"C:\Users\kirdr\.spyder-py3\TimeDataSets\Hourly-train-processed.csv")

predictor.plot(test_data, predictions, quantile_levels=[0.1, 0.9], max_history_length=200, max_num_item_ids=4)