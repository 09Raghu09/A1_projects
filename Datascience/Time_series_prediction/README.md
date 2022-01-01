# Forecast the outbreak

Machine learning models are calibrated using publicly available data sources like the WHO health
report. Time series forecasting can be framed as a supervised learning problem. Other than model-

The forecasting is performed using pythons numpy and sklearn libraries. At first, the data is
downloaded. Since no data points are missing no preprocessing is performed with the exception of
converting integers into date times and reorganizing dataframes.

This project mainly focus on forecasting the covid outbreak and extract meaningful statistics and interesting characteristics from the time series
data structure. Thus, investigating the stationary and seasonality. It is said to be stationary if it’s mean and variance does not
change over time, while seasonality can be identified depending on the data, which refers to periodic
fluctuations of the values. The second major intention is to perform forecasting. Here, a model is
applied to the data to predict future values based on past observations.

