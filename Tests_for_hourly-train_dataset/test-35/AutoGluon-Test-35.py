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
    presets="medium_quality",
    time_limit=5400,
    num_val_windows=4,
    enable_ensemble=False,
    val_step_size =2,
    hyperparameters={
      "Theta": {},
      "TemporalFusionTransformer": {},
      "DirectTabular": {},
      "RecursiveTabular": {},
      "ETS": {},
   },
)

predictions = predictor.predict(train_data)
predictions.head()

# TimeSeriesDataFrame can also be loaded directly from a file
test_data = TimeSeriesDataFrame.from_path(r"C:\Users\kirdr\.spyder-py3\TimeDataSets\Hourly-train-processed.csv")

plt.figure(figsize=(20, 3))

item_id = "H1"
y_past = train_data.loc[item_id]["target"]
y_pred = predictions.loc[item_id]
y_test = test_data.loc[item_id]["target"][-48:]

plt.plot(y_past[-200:], label="Past time series values")
plt.plot(y_pred["mean"], label="Mean forecast")
plt.plot(y_test, label="Future time series values")

plt.fill_between(
    y_pred.index, y_pred["0.1"], y_pred["0.9"], color="red", alpha=0.1, label=f"10%-90% confidence interval"
)
plt.legend();