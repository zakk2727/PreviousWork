
# import packages for analysis and modeling
import pandas as pd #data frame operations
import numpy as np #arrays and math functions
from scipy.stats import uniform #for training and test splits
import statsmodels.api as smf #R-like model specification
import matplotlib.pyplot as plt #2D plotting
from statsmodels.tsa.stattools import adfuller
# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

# read in Real Estate data and create data frame
RealEstate = pd.read_csv("Zip_zhvi_sfr.csv")
RealEstateDF = pd.DataFrame(RealEstate)

# CLEAN RealEstate DF
ColumnsToDrop = ['RegionID','SizeRank','RegionType','StateName','2025-01-31','2025-02-28','2025-03-31','2025-04-30']
RealEstateDF.drop(columns=ColumnsToDrop,inplace=True)

RealEstateDFNA = RealEstateDF[RealEstateDF['2000-01-31'].isna()]
RealEstateDF.dropna(inplace=True)

#print(RealEstateDF.columns)

RealEstateDF['2024Average'] = RealEstateDF.iloc[:,-12:].mean(skipna=True, axis=1)
Q60 = RealEstateDF['2024Average'].quantile(q = 0.60)
Q35 = RealEstateDF['2024Average'].quantile(q = 0.35)

#print(RealEstateDF)

RealEstateDF['ROI'] = (RealEstateDF['2024-12-31']/RealEstateDF['2000-01-31']) - 1
RealEstateDF['STD'] = RealEstateDF.loc[:,'2000-01-31':'2024-12-31'].std(skipna=True, axis=1)
RealEstateDF['MEAN'] = RealEstateDF.loc[:,'2000-01-31':'2024-12-31'].mean(skipna=True, axis=1)
RealEstateDF['CV'] = (RealEstateDF['STD']/RealEstateDF['MEAN'])
UpperCV = RealEstateDF.CV.quantile(.5)

TempDF = RealEstateDF.copy()
TempColumnsToDrop = ['RegionName','State','City','Metro','CountyName','ROI','STD','MEAN','CV','2024Average']
TempDF.drop(columns=TempColumnsToDrop,inplace=True)

DateColumnNames = list(set(TempDF.columns))

RealEstateDF02 = RealEstateDF.copy()

RealEstateDF02 = pd.melt(RealEstateDF02, id_vars = ['RegionName','State','City','Metro','CountyName','ROI','STD','MEAN','CV','2024Average'], value_vars = DateColumnNames , var_name = 'Date').sort_values(['RegionName','Date'])
RealEstateDF02 = RealEstateDF02.reset_index(drop = True)

RealEstateDF02['Date'] = pd.to_datetime(RealEstateDF02['Date'])

#print(RealEstateDF02)

RealEstateDF['ROI'].hist()
#plt.show()

RealEstateDF = RealEstateDF[(RealEstateDF['2024Average'] < Q60) & (RealEstateDF['2024Average'] > Q35)]
#print(RealEstateDF)
print(RealEstateDF[RealEstateDF['CV'] < UpperCV].sort_values('ROI',ascending = False))
RealEstateBest3ROI_DF = RealEstateDF[RealEstateDF['CV'] < UpperCV].sort_values('ROI',ascending = False)[:20]
print(RealEstateBest3ROI_DF[['RegionName','State','City','CountyName','ROI','CV']].sort_values('CV',ascending = False))

TempTop3DF = RealEstateBest3ROI_DF.copy()
TempTop3ColumnsToDrop = ['RegionName','State','City','Metro','CountyName','ROI']
TempTop3DF.drop(columns=TempTop3ColumnsToDrop,inplace=True)

RealEstateBest3ROI_DF = pd.melt(RealEstateBest3ROI_DF, id_vars = ['RegionName','State','City','Metro','CountyName','ROI'], value_vars = DateColumnNames , var_name = 'Date').sort_values(['RegionName','Date'])
RealEstateBest3ROI_DF = RealEstateBest3ROI_DF.reset_index(drop = True)

RealEstateBest3ROI_DF['Date'] = pd.to_datetime(RealEstateBest3ROI_DF['Date'])
RealEstateBest3ROI_DF.pivot(index="Date",columns="City",values="value").plot()
#plt.show()

Philadelphia_DF = RealEstateBest3ROI_DF[RealEstateBest3ROI_DF.City == "Philadelphia"]
Philadelphia_DF = Philadelphia_DF.set_index('Date')
Beachwood_DF = RealEstateBest3ROI_DF[RealEstateBest3ROI_DF.City == "Beachwood"]
Beachwood_DF = Beachwood_DF.set_index('Date')
Whiting_DF = RealEstateBest3ROI_DF[RealEstateBest3ROI_DF.City == "Whiting"]
Whiting_DF = Whiting_DF.set_index('Date')
Rio_DF = RealEstateBest3ROI_DF[RealEstateBest3ROI_DF.City == "Rio Grande"]
Rio_DF = Rio_DF.set_index('Date')
Putnam_DF = RealEstateBest3ROI_DF[RealEstateBest3ROI_DF.City == "Putnam Station"]
Putnam_DF = Putnam_DF.set_index('Date')

CityColumnsToDrop = ['RegionName','State','City','Metro','CountyName','ROI']
Philadelphia_DF.drop(columns=CityColumnsToDrop,inplace=True)
Beachwood_DF.drop(columns=CityColumnsToDrop,inplace=True)
Whiting_DF.drop(columns=CityColumnsToDrop,inplace=True)
Rio_DF.drop(columns=CityColumnsToDrop,inplace=True)
Putnam_DF.drop(columns=CityColumnsToDrop,inplace=True)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
Scaler = MinMaxScaler(feature_range=(-1, 1))

Philadelphia_DF['DifferencedValue'] = Philadelphia_DF['value'] - Philadelphia_DF['value'].shift(1)
Beachwood_DF['DifferencedValue'] = Beachwood_DF['value'] - Beachwood_DF['value'].shift(1)
Whiting_DF['DifferencedValue'] = Whiting_DF['value'] - Whiting_DF['value'].shift(1)
Rio_DF['DifferencedValue'] = Rio_DF['value'] - Rio_DF['value'].shift(1)
Putnam_DF['DifferencedValue'] = Putnam_DF['value'] - Putnam_DF['value'].shift(1)

Philadelphia_DF['DifferencedValue'] = Philadelphia_DF['DifferencedValue'].fillna(0)
Beachwood_DF['DifferencedValue'] = Beachwood_DF['DifferencedValue'].fillna(0)
Whiting_DF['DifferencedValue'] = Whiting_DF['DifferencedValue'].fillna(0)
Rio_DF['DifferencedValue'] = Rio_DF['DifferencedValue'].fillna(0)
Putnam_DF['DifferencedValue'] = Putnam_DF['DifferencedValue'].fillna(0)

Philadelphia_DF.drop(columns='value',inplace=True)
Beachwood_DF.drop(columns='value',inplace=True)
Whiting_DF.drop(columns='value',inplace=True)
Rio_DF.drop(columns='value',inplace=True)
Putnam_DF.drop(columns='value',inplace=True)

Philadelphia_DF['ScaledDiffValue'] = Scaler.fit_transform(Philadelphia_DF)
Beachwood_DF['ScaledDiffValue'] = Scaler.fit_transform(Beachwood_DF)
Whiting_DF['ScaledDiffValue'] = Scaler.fit_transform(Whiting_DF)
Rio_DF['ScaledDiffValue'] = Scaler.fit_transform(Rio_DF)
Putnam_DF['ScaledDiffValue'] = Scaler.fit_transform(Putnam_DF)

Philadelphia_DF['RegionName'] = 19127
Beachwood_DF['RegionName'] = 8722
Whiting_DF['RegionName'] = 8733
Rio_DF['RegionName'] = 8242
Putnam_DF['RegionName'] = 12861

Philadelphia_DF = Philadelphia_DF.reset_index(drop = False)
Beachwood_DF = Beachwood_DF.reset_index(drop = False)
Whiting_DF = Whiting_DF.reset_index(drop = False)
Rio_DF = Rio_DF.reset_index(drop = False)
Putnam_DF = Putnam_DF.reset_index(drop = False)

ConcatMonthReturn_DF = pd.concat([Philadelphia_DF,Beachwood_DF,Whiting_DF,Rio_DF,Putnam_DF])
RealEstateBest3ROI_DF = pd.merge(left=RealEstateBest3ROI_DF,right=ConcatMonthReturn_DF[['RegionName','Date','ScaledDiffValue']], on=['RegionName','Date'], how='left')

RealEstateBest3ROI_DFNA = RealEstateBest3ROI_DF[RealEstateBest3ROI_DF['ScaledDiffValue'].isna()]
RealEstateBest3ROI_DF.dropna(inplace=True)

RealEstateBest3ROI_DF.pivot(index="Date",columns="City",values="ScaledDiffValue").plot()
#plt.show()

X = RealEstateBest3ROI_DF.ScaledDiffValue[RealEstateBest3ROI_DF.City == "Philadelphia"]
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

X = RealEstateBest3ROI_DF.ScaledDiffValue[RealEstateBest3ROI_DF.City == "Beachwood"]
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

X = RealEstateBest3ROI_DF.ScaledDiffValue[RealEstateBest3ROI_DF.City == "Whiting"]
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

X = RealEstateBest3ROI_DF.ScaledDiffValue[RealEstateBest3ROI_DF.City == "Rio Grande"]
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

X = RealEstateBest3ROI_DF.ScaledDiffValue[RealEstateBest3ROI_DF.City == "Putnam Station"]
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

### ARIMA MODELS ###

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',FutureWarning)
#from pmdarima.arima import auto_arima
#warnings.warn(ARIMA_DEPRECATION_WARN, FutureWarning)
from statsforecast import StatsForecast
from statsforecast.arima import arima_string
from statsforecast.models import AutoARIMA

#pd.plotting.autocorrelation_plot(RealEstateBest3ROI_DF.MonthlyReturn[RealEstateBest3ROI_DF.City == "Beachwood"])
#plt.show()

#print(RealEstateBest3ROI_DF[RealEstateBest3ROI_DF.isna().any(axis=1)])

"""

CityDFColumnsToDrop = ['Date','ScaledDiffValue','RegionName','DifferencedValue']
P_DF = Philadelphia_DF.copy()
P_DF['ds'] = pd.to_datetime(P_DF['Date'])
P_DF['y'] = P_DF['ScaledDiffValue']
P_DF['unique_id'] = "1"
P_DF.drop(columns=CityDFColumnsToDrop,inplace=True)
P_DF = P_DF.reset_index(drop = True)

B_DF = Beachwood_DF.copy()
B_DF['ds'] = pd.to_datetime(B_DF['Date'])
B_DF['y'] = B_DF['ScaledDiffValue']
B_DF['unique_id'] = "1"
B_DF.drop(columns=CityDFColumnsToDrop,inplace=True)
B_DF = B_DF.reset_index(drop = True)

W_DF = Whiting_DF.copy()
W_DF['ds'] = pd.to_datetime(W_DF['Date'])
W_DF['y'] = W_DF['ScaledDiffValue']
W_DF['unique_id'] = "1"
W_DF.drop(columns=CityDFColumnsToDrop,inplace=True)
W_DF = W_DF.reset_index(drop = True)

RG_DF = Rio_DF.copy()
RG_DF['ds'] = pd.to_datetime(RG_DF['Date'])
RG_DF['y'] = RG_DF['ScaledDiffValue']
RG_DF['unique_id'] = "1"
RG_DF.drop(columns=CityDFColumnsToDrop,inplace=True)
RG_DF = RG_DF.reset_index(drop = True)

PS_DF = Putnam_DF.copy()
PS_DF['ds'] = pd.to_datetime(PS_DF['Date'])
PS_DF['y'] = PS_DF['ScaledDiffValue']
PS_DF['unique_id'] = "1"
PS_DF.drop(columns=CityDFColumnsToDrop,inplace=True)
PS_DF = PS_DF.reset_index(drop = True)

PAutoModel = AutoARIMA(d=1,D=1,max_p=5,max_q=5,max_P=5,max_Q=5,max_order=5,max_d=5,max_D=5,start_p=0,start_q=0,start_P=0,start_Q=0,seasonal=True,stepwise=True,nmodels=100,trace=True,season_length=12)
PAutoSF = StatsForecast(models=[PAutoModel],freq='M',n_jobs=-1)
PAutoSFModel = PAutoSF.fit(df=P_DF)

BAutoModel = AutoARIMA(d=1,D=1,max_p=5,max_q=5,max_P=5,max_Q=5,max_order=5,max_d=5,max_D=5,start_p=0,start_q=0,start_P=0,start_Q=0,seasonal=True,stepwise=True,nmodels=100,trace=True,season_length=12)
BAutoSF = StatsForecast(models=[BAutoModel],freq='M',n_jobs=-1)
BAutoSFModel = BAutoSF.fit(df=B_DF)

WAutoModel = AutoARIMA(d=1,D=1,max_p=5,max_q=5,max_P=5,max_Q=5,max_order=5,max_d=5,max_D=5,start_p=0,start_q=0,start_P=0,start_Q=0,seasonal=True,stepwise=True,nmodels=100,trace=True,season_length=12)
WAutoSF = StatsForecast(models=[WAutoModel],freq='M',n_jobs=-1)
WAutoSFModel = WAutoSF.fit(df=W_DF)

RGAutoModel = AutoARIMA(d=1,D=1,max_p=5,max_q=5,max_P=5,max_Q=5,max_order=5,max_d=5,max_D=5,start_p=0,start_q=0,start_P=0,start_Q=0,seasonal=True,stepwise=True,nmodels=100,trace=True,season_length=12)
RGAutoSF = StatsForecast(models=[RGAutoModel],freq='M',n_jobs=-1)
RGAutoSFModel = RGAutoSF.fit(df=RG_DF)

PSAutoModel = AutoARIMA(d=0,D=0,max_p=5,max_q=5,max_P=5,max_Q=5,max_order=5,max_d=5,max_D=5,start_p=0,start_q=0,start_P=0,start_Q=0,seasonal=True,stepwise=True,nmodels=100,trace=True,season_length=12)
PSAutoSF = StatsForecast(models=[PSAutoModel],freq='M',n_jobs=-1)
PSAutoSFModel = PSAutoSF.fit(df=PS_DF)

print(arima_string(PAutoSFModel.fitted_[0,0].model_))
print(arima_string(BAutoSFModel.fitted_[0,0].model_))
print(arima_string(WAutoSFModel.fitted_[0,0].model_))
print(arima_string(RGAutoSFModel.fitted_[0,0].model_))
print(arima_string(PSAutoSFModel.fitted_[0,0].model_))

"""

RealEstateBest3ROI_DF = RealEstateBest3ROI_DF.set_index('Date')

PhillyARIMAModel = ARIMA(RealEstateBest3ROI_DF.ScaledDiffValue[RealEstateBest3ROI_DF.City == "Philadelphia"], order=(2,1,3),seasonal_order=(3,1,1,12),freq='M')
PhillyARIMAModelFit = PhillyARIMAModel.fit()
print(PhillyARIMAModelFit.summary())

BeachwoodARIMAModel = ARIMA(RealEstateBest3ROI_DF.ScaledDiffValue[RealEstateBest3ROI_DF.City == "Beachwood"], order=(3,1,3),seasonal_order=(2,1,1,12),freq='M')
BeachwoodARIMAModelFit = BeachwoodARIMAModel.fit()
print(BeachwoodARIMAModelFit.summary())

WhitingARIMAModel = ARIMA(RealEstateBest3ROI_DF.ScaledDiffValue[RealEstateBest3ROI_DF.City == "Whiting"], order=(5,1,1),seasonal_order=(3,1,1,12),freq='M')
WhitingARIMAModelFit = WhitingARIMAModel.fit()
print(WhitingARIMAModelFit.summary())

RioGrandeARIMAModel = ARIMA(RealEstateBest3ROI_DF.ScaledDiffValue[RealEstateBest3ROI_DF.City == "Rio Grande"], order=(2,1,1),seasonal_order=(5,1,3,12),freq='M')
RioGrandeARIMAModelFit = RioGrandeARIMAModel.fit()
print(RioGrandeARIMAModelFit.summary())

PSARIMAModel = ARIMA(RealEstateBest3ROI_DF.ScaledDiffValue[RealEstateBest3ROI_DF.City == "Putnam Station"], order=(5,0,0),seasonal_order=(1,0,0,12),freq='M')
PSARIMAModelFit = PSARIMAModel.fit()
print(PSARIMAModelFit.summary())

#####

PhillyModel = SARIMAX(RealEstateBest3ROI_DF.ScaledDiffValue[RealEstateBest3ROI_DF.City == "Philadelphia"], order=(2,1,3),seasonal_order=(3,1,1,12),freq='M')
PhillyModelFit = PhillyModel.fit(disp=0)
print(PhillyModelFit.summary())

BeachwoodModel = SARIMAX(RealEstateBest3ROI_DF.ScaledDiffValue[RealEstateBest3ROI_DF.City == "Beachwood"], order=(3,1,3),seasonal_order=(2,1,1,12),freq='M')
BeachwoodModelFit = BeachwoodModel.fit(disp=0)
print(BeachwoodModelFit.summary())

WhitingModel = SARIMAX(RealEstateBest3ROI_DF.ScaledDiffValue[RealEstateBest3ROI_DF.City == "Whiting"], order=(5,1,1),seasonal_order=(3,1,1,12),freq='M')
WhitingModelFit =WhitingModel.fit(disp=0)
print(WhitingModelFit.summary())

RioGrandeModel = SARIMAX(RealEstateBest3ROI_DF.ScaledDiffValue[RealEstateBest3ROI_DF.City == "Rio Grande"], order=(2,1,1),seasonal_order=(5,1,3,12),freq='M')
RioGrandeModelFit = RioGrandeModel.fit(disp=0)
print(RioGrandeModelFit.summary())

PSModel = SARIMAX(RealEstateBest3ROI_DF.ScaledDiffValue[RealEstateBest3ROI_DF.City == "Putnam Station"], order=(5,0,0),seasonal_order=(1,0,0,12))
PSModelFit = PSModel.fit(disp=0)
print(PSModelFit.summary())

POutput = pd.DataFrame(PhillyModelFit.predict(start='2025-01-31',end='2025-12-31',dynamic=True))
BOutput =  pd.DataFrame(BeachwoodModelFit.predict(start='2025-01-31',end='2025-12-31',dynamic=True))
WOutput =  pd.DataFrame(WhitingModelFit.predict(start='2025-01-31',end='2025-12-31',dynamic=True))
RGOutput =  pd.DataFrame(RioGrandeModelFit.predict(start='2025-01-31',end='2025-12-31',dynamic=True))
PSOutput =  pd.DataFrame(PSModelFit.predict(start='2025-01-31',end='2025-12-31',dynamic=True))

POutput['InverseScaleValue'] = Scaler.inverse_transform(POutput)
BOutput['InverseScaleValue'] = Scaler.inverse_transform(BOutput)
WOutput['InverseScaleValue'] = Scaler.inverse_transform(WOutput)
RGOutput['InverseScaleValue'] = Scaler.inverse_transform(RGOutput)
PSOutput['InverseScaleValue'] = Scaler.inverse_transform(PSOutput)

PLastOb = RealEstateBest3ROI_DF.value[RealEstateBest3ROI_DF.City == "Philadelphia"]['2024-12-31']
BLastOb = RealEstateBest3ROI_DF.value[RealEstateBest3ROI_DF.City == "Beachwood"]['2024-12-31']
WLastOb = RealEstateBest3ROI_DF.value[RealEstateBest3ROI_DF.City == "Whiting"]['2024-12-31']
RGLastOb = RealEstateBest3ROI_DF.value[RealEstateBest3ROI_DF.City == "Rio Grande"]['2024-12-31']
PSLastOb = RealEstateBest3ROI_DF.value[RealEstateBest3ROI_DF.City == "Putnam Station"]['2024-12-31']

print(PLastOb)

POutput['FinalValue'] = 0
BOutput['FinalValue'] = 0
WOutput['FinalValue'] = 0
RGOutput['FinalValue'] = 0
PSOutput['FinalValue'] = 0

for i in range(len(POutput)):
	if (i == 0):
		POutput['FinalValue'][i] = POutput['InverseScaleValue'][i] + PLastOb
	else:
		POutput['FinalValue'][i] = POutput['FinalValue'][i-1] + POutput['InverseScaleValue'][i]

for i in range(len(BOutput)):
	if (i == 0):
		BOutput['FinalValue'][i] = BOutput['InverseScaleValue'][i] + BLastOb
	else:
		BOutput['FinalValue'][i] = BOutput['FinalValue'][i-1] + BOutput['InverseScaleValue'][i]

for i in range(len(WOutput)):
	if (i == 0):
		WOutput['FinalValue'][i] = WOutput['InverseScaleValue'][i] + WLastOb
	else:
		WOutput['FinalValue'][i] = WOutput['FinalValue'][i-1] + WOutput['InverseScaleValue'][i]

for i in range(len(RGOutput)):
	if (i == 0):
		RGOutput['FinalValue'][i] = RGOutput['InverseScaleValue'][i] + RGLastOb
	else:
		RGOutput['FinalValue'][i] = RGOutput['FinalValue'][i-1] + RGOutput['InverseScaleValue'][i]

for i in range(len(PSOutput)):
	if (i == 0):
		PSOutput['FinalValue'][i] = PSOutput['InverseScaleValue'][i] + PSLastOb
	else:
		PSOutput['FinalValue'][i] = PSOutput['FinalValue'][i-1] + PSOutput['InverseScaleValue'][i]

POutput = POutput.reset_index(drop = False)
POutput.columns = ['Date','ScaledDiffValue','InverseScaleValue','PhiladelphiaPredValue']
POutput['City'] = "Philadelphia"

BOutput = BOutput.reset_index(drop = False)
BOutput.columns = ['Date','ScaledDiffValue','InverseScaleValue','BeachwoodPredValue']
BOutput['City'] = "Beachwood"

WOutput = WOutput.reset_index(drop = False)
WOutput.columns = ['Date','ScaledDiffValue','InverseScaleValue','WhitingPredValue']
WOutput['City'] = "Whiting"

RGOutput = RGOutput.reset_index(drop = False)
RGOutput.columns = ['Date','ScaledDiffValue','InverseScaleValue','RioGrandePredValue']
RGOutput['City'] = "Rio Grande"

PSOutput = PSOutput.reset_index(drop = False)
PSOutput.columns = ['Date','ScaledDiffValue','InverseScaleValue','PSPredValue']
PSOutput['City'] = "Putnam Station"

ResultsDF = pd.merge(left=POutput[['Date','PhiladelphiaPredValue']],right=BOutput[['Date','BeachwoodPredValue']], on=['Date'], how='left')
ResultsDF = pd.merge(left=ResultsDF,right=WOutput[['Date','WhitingPredValue']], on=['Date'], how='left')
ResultsDF = pd.merge(left=ResultsDF,right=RGOutput[['Date','RioGrandePredValue']], on=['Date'], how='left')
ResultsDF = pd.merge(left=ResultsDF,right=PSOutput[['Date','PSPredValue']], on=['Date'], how='left')

ResultsDF = ResultsDF.reset_index(drop = True)
print(ResultsDF)

RealEstateBest3ROI_DF = RealEstateBest3ROI_DF.reset_index(drop = False)

POriginal = RealEstateBest3ROI_DF[['Date','value']][RealEstateBest3ROI_DF.City == "Philadelphia"]
POriginal.columns = ['Date','PhiladelphiaValue']
BOriginal = RealEstateBest3ROI_DF[['Date','value']][RealEstateBest3ROI_DF.City == "Beachwood"]
BOriginal.columns = ['Date','BeachwoodValue']
WOriginal = RealEstateBest3ROI_DF[['Date','value']][RealEstateBest3ROI_DF.City == "Whiting"]
WOriginal.columns = ['Date','WhitingValue']
RGOriginal = RealEstateBest3ROI_DF[['Date','value']][RealEstateBest3ROI_DF.City == "Rio Grande"]
RGOriginal.columns = ['Date','RioGrandeValue']
PSOriginal = RealEstateBest3ROI_DF[['Date','value']][RealEstateBest3ROI_DF.City == "Putnam Station"]
PSOriginal.columns = ['Date','PutnamStationValue']

OriginalValuesDF = pd.merge(left=POriginal,right=BOriginal[['Date','BeachwoodValue']], on=['Date'], how='left')
OriginalValuesDF = pd.merge(left=OriginalValuesDF,right=WOriginal[['Date','WhitingValue']], on=['Date'], how='left')
OriginalValuesDF = pd.merge(left=OriginalValuesDF,right=RGOriginal[['Date','RioGrandeValue']], on=['Date'], how='left')
OriginalValuesDF = pd.merge(left=OriginalValuesDF,right=PSOriginal[['Date','PutnamStationValue']], on=['Date'], how='left')
print(OriginalValuesDF)


plt.plot(OriginalValuesDF['Date'], OriginalValuesDF['PhiladelphiaValue'], label = 'Philadelphia, PA')
plt.plot(OriginalValuesDF['Date'], OriginalValuesDF['BeachwoodValue'], label = 'Beachwood, NJ')
plt.plot(OriginalValuesDF['Date'], OriginalValuesDF['WhitingValue'], label = 'Whiting, NJ')
plt.plot(OriginalValuesDF['Date'], OriginalValuesDF['RioGrandeValue'], label = 'Rio Grande, NJ')
plt.plot(OriginalValuesDF['Date'], OriginalValuesDF['PutnamStationValue'], label = 'Putnam Station, NY')
plt.plot(ResultsDF['Date'], ResultsDF['PhiladelphiaPredValue'], label = 'Forecasted Values',color='black')
plt.plot(ResultsDF['Date'], ResultsDF['BeachwoodPredValue'],color='black')
plt.plot(ResultsDF['Date'], ResultsDF['WhitingPredValue'],color='black')
plt.plot(ResultsDF['Date'], ResultsDF['RioGrandePredValue'],color='black')
plt.plot(ResultsDF['Date'], ResultsDF['PSPredValue'],color='black')
plt.legend()
plt.show()

plt.plot(ResultsDF['Date'], ResultsDF['PhiladelphiaPredValue'], label = 'Philadelphia, PA')
plt.plot(ResultsDF['Date'], ResultsDF['BeachwoodPredValue'], label = 'Beachwood, NJ')
plt.plot(ResultsDF['Date'], ResultsDF['WhitingPredValue'], label = 'Whiting, NJ')
plt.plot(ResultsDF['Date'], ResultsDF['RioGrandePredValue'], label = 'Rio Grande, NJ')
plt.plot(ResultsDF['Date'], ResultsDF['PSPredValue'], label = 'Putnam Station, NY')
plt.legend()
plt.show()

ResultsDF = ResultsDF.set_index('Date')
OriginalValuesDF = OriginalValuesDF.set_index('Date')

PPercentChange =  (ResultsDF['PhiladelphiaPredValue']['2025-12-31'] / OriginalValuesDF['PhiladelphiaValue']['2024-12-31']) - 1
BPercentChange =  (ResultsDF['BeachwoodPredValue']['2025-12-31'] / OriginalValuesDF['BeachwoodValue']['2024-12-31']) - 1
WPercentChange =  (ResultsDF['WhitingPredValue']['2025-12-31'] / OriginalValuesDF['WhitingValue']['2024-12-31']) - 1
RGPercentChange =  (ResultsDF['RioGrandePredValue']['2025-12-31'] / OriginalValuesDF['RioGrandeValue']['2024-12-31']) - 1
PSPercentChange =  (ResultsDF['PSPredValue']['2025-12-31'] / OriginalValuesDF['PutnamStationValue']['2024-12-31']) - 1

print(f'Total expected increase in median home value in 1 year for Philadelphia, PA: {round(PPercentChange*100,2)}%')
print(f'Total expected increase in median home value in 1 year for Beachwood, NJ: {round(BPercentChange*100,2)}%')
print(f'Total expected increase in median home value in 1 year for Whiting, NJ: {round(WPercentChange*100,2)}%')
print(f'Total expected increase in median home value in 1 year for Rio Grande, NJ: {round(RGPercentChange*100,2)}%')
print(f'Total expected increase in median home value in 1 year for Putnam Station, NY: {round(PSPercentChange*100,2)}%')

