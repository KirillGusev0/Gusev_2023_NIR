# -*- coding: utf-8 -*-
"""
Created on Fri May 10 20:49:58 2024

@author: kirdr
"""

import matplotlib.pyplot as plt
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import os
import sys

def predictions_with_chosen_parametrs(data_file,data_list): 
   df = pd.read_csv(data_file)
   df.head()
   
   # Создание пути для сохранения файла лога
   log_dir = "log_and_grafs"
   
    
   train_data = TimeSeriesDataFrame.from_data_frame(
       df,
       id_column="item_id",
       timestamp_column="timestamp"
   )
   train_data.head()

   predictor = TimeSeriesPredictor(
       prediction_length=data_list['prediction_length'],
       target="target",
       eval_metric=data_list['metrics'],
   )

   predictor.fit(
       train_data,
       presets=data_list['presets'],
       time_limit=data_list['time_limit'],
       num_val_windows=data_list['num_val_windows'],
       enable_ensemble=data_list['enable_ensemble'],
       val_step_size=data_list['val_step_size'],
       hyperparameters=data_list['hyperparameters'],
   )

   predictions = predictor.predict(train_data)
   predictions.head()

   # TimeSeriesDataFrame can also be loaded directly from a file
   test_data = TimeSeriesDataFrame.from_path(data_file)

   item_id = "H1"

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

   # Путь для сохранения графика
   plot_file = os.path.join(log_dir, f"plot_1_{os.path.basename(data_file)}.png")

   # Сохранение графика
   plt.savefig(plot_file)

   plt.show()

   # Построение графика без зеленого графика
   plt.figure(figsize=(20, 3))
   plt.plot(y_past[-200:], label="Past time series values")
   plt.plot(y_pred.index, y_pred["mean"], label="Mean forecast", color="red")  # Красный график для прогноза
   plt.fill_between(
       y_pred.index, y_pred["0.1"], y_pred["0.9"], color="red", alpha=0.1, label=f"10%-90% confidence interval"
   )
   plt.legend()

   # Путь для сохранения графика
   plot_file = os.path.join(log_dir, f"plot_2_{os.path.basename(data_file)}.png")

   # Сохранение графика
   plt.savefig(plot_file)

   # Отображение графика
   plt.show()
   return 0