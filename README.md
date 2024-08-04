<p> The repository motivates LSTM-based univariate multistep forecasting as an imputation 
technique for missing sequential data. The temporal and between-feature correlation in 
the absence, which often characterizes the missing data of sensors, undermines the ability 
of traditional imputation techniques. <p/>

<p> For this purpose, the imputation accuracy of two LSTM-based architectures is compared with 
linear and spline imputation upon a dataset of meteorological data from Berlin in 2023. 
The data is provided by the Deutscher Wetterdienst (German Weather Service). </p>

<h2> Structure of the repository: </h2>

1. <em>datasets</em>:  hourly measurements of X meteorological characteristic for Berlin 2023, w/ and without imputation
2. <em>utilities</em>: required functions
3. <em>EMDA.ipynb</em>: Exploratory missing data analysis 
4. <em>imputation.ipynb</em>: imputation based upon EDA results
5. <em>LSTM_results.ipynb</em>: comparison of different imputation techniques
 
<h2> Datasource: </h2>


The usage of the meteorological data is regulated by the "Creative
Commons BY 4.0" (CC BY 4.0) and covers the following data sets:

- Hourly station observations of 2 m air temperature and humidity
for Germany, Version v24.03
- Hourly station observations of precipitation for Germany, Version v24.03
- Hourly station observations of pressure for Germany, Version
v24.03
- Hourly station observations of solar incoming (total/diffuse)
and longwave downward radiation for Germany, Version v24.03
- Hourly mean value from station observations of wind speed and
wind direction for Germany, Version v24.03

<h2> Ackknowledgement: </h2>

The architecture for multi-step forecast, applied for missing data
imputation, has been altered from the tutorial by [Brownlee (2020)](https://machinelearningmastery.com/multi-step-time-series-forecasting-with-machine-learningmodels-for-household-electricity-consumption/).

The original owner of the data and code used
in this thesis retains ownership of the data and code.


Brownlee, J. (2020). Multi-step time series forecasting with machine learning
for electricity usage. 
