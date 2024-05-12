# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 17:24:19 2023

@author: kirdr
"""
import matplotlib.pyplot as plt
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

df = pd.read_csv(r"C:\Users\kirdr\.spyder-py3\TimeDataSets\Weekly-train-processed.csv")
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
    time_limit=1800,
    num_val_windows=4,
    enable_ensemble=False,
    val_step_size =2,
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
test_data = TimeSeriesDataFrame.from_path(r"C:\Users\kirdr\.spyder-py3\TimeDataSets\Weekly-train-processed.csv")

item_id = "W1" 


# Получение данных для графика
y_past = train_data.loc[item_id]["target"]
y_pred = predictions.loc[item_id]
y_test = test_data.loc[item_id]["target"][-48:]

# Построение графика с красным и зеленым графиком
plt.figure(figsize=(20, 3))
plt.plot(y_past[-200:], label="Past time series values")
plt.plot(y_pred.index, y_pred["mean"], label="Mean forecast", color="red")  # Красный график для прогноза
plt.fill_between(
    y_pred.index, y_pred["0.1"], y_pred["0.9"], color="red", alpha=0.1, label=f"10%-90% confidence interval"
)
plt.plot(y_pred.index, y_test, label="Future time series values", color="green")  # Зеленый график для фактических значений
plt.legend()
plt.show()

# Построение графика без зеленого графика
plt.figure(figsize=(20, 3))
plt.plot(y_past[-200:], label="Past time series values")
plt.plot(y_pred.index, y_pred["mean"], label="Mean forecast", color="red")  # Красный график для прогноза
plt.fill_between(
    y_pred.index, y_pred["0.1"], y_pred["0.9"], color="red", alpha=0.1, label=f"10%-90% confidence interval"
)
plt.legend()
plt.show()