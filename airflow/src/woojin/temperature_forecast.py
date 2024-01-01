import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import prophet
from prophet import Prophet
import json


import warnings
warnings.filterwarnings('ignore')


# time을 시계열 데이터로 변경
def timeseries(X):

    X['ds'] = pd.to_datetime(X['ds'], format='%Y-%m-%d %H:%M:%S')
    X.set_index('ds',drop=True,inplace=True)
    X = X.sort_index()

    return X





# 전체 평균 온도 구하기


# 상한선을 구하는 함수 정의
def calculate_upper_bound(data, threshold=1.5):
    q3 = np.percentile(data, 75)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    upper_bound = q3 + (threshold * iqr)
    return upper_bound.round()



def data_forecast(X , model):
  study_df = pd.DataFrame()
  study_df['ds'] = X.reset_index(drop=False)['ds']
  study_df['y'] = X.reset_index(drop=False)['y']

  model.fit(study_df)

  #future = model.make_future_dataframe(periods=24,freq='H')
  future = model.make_future_dataframe(periods=24*60, freq='T')

  forecast = model.predict(future) # 예측 결과 생성

  return forecast



def inference(actual, forecast):
  start_time = actual.index[-1]  # 실제값의 마지막 시간
  end_time = start_time + pd.DateOffset(days=1)  # 24시간 후의 시간 범위
  time_range = pd.date_range(start=start_time, end=end_time, freq='T')

  result_actual = actual.copy()
  result_actual.reset_index(inplace=True)
  result_actual = result_actual.rename(columns={'index': 'TimeStamp'})

  result_forecast = forecast.loc[forecast['ds'].isin(time_range)].copy()
  result_forecast = result_forecast.iloc[1:].copy()
  result_forecast.reset_index(drop=True, inplace=True)
  result_forecast = result_forecast.drop(['trend','yhat_lower','yhat_upper','trend_lower','trend_upper','additive_terms','additive_terms_lower','additive_terms_upper','multiplicative_terms',
                      'multiplicative_terms_lower','multiplicative_terms_upper'],axis=1)

  result_forecast = result_forecast.rename(columns={'yhat': 'y'})

  result = pd.concat([result_actual, result_forecast], axis=0)
  result = result.rename(columns={'ds': 'TimeStamp', 'y': 'Forecast'})
  result.reset_index(drop=True, inplace=True)

  return result



def split_data_by_day_and_interval(data, interval):

    daily_data = data.groupby(pd.Grouper(key='TimeStamp', freq='D'))

    json_data_today= {}
    # for date, daily_group in daily_data:
    #     json_data[str(date.date())] = {}
    #     for column in daily_group.columns[1:]:
    #         time_range = pd.date_range(date, periods=24*60//interval, freq=f'{interval}T')
    #         subset_data = daily_group[[column]].values.flatten().tolist()
    #         json_data[str(date.date())][column] = {str(t.time()): val for t, val in zip(time_range, subset_data)}

    for date, daily_group in daily_data:
        print("sdfsf",date)
        for column in daily_group.columns[1:]:
            # json_data[column] = {}
            time_range = pd.date_range(date, periods=24*60//interval, freq=f'{interval}T')
            subset_data = daily_group[[column]].values.flatten().tolist()
            json_data_today[column] = {(str(date.date())+'_'+str(t.time())): val for t, val in zip(time_range, subset_data)}
        # print(json_data)
        break
    print(json_data_today.keys)

    json_data_tommo= {}
    for date, daily_group in daily_data:
        print("sdfsf",date)
        for column in daily_group.columns[1:]:
            # json_data[column] = {}
            time_range = pd.date_range(date, periods=24*60//interval, freq=f'{interval}T')
            subset_data = daily_group[[column]].values.flatten().tolist()
            json_data_tommo[column] = {(str(date.date())+'_'+str(t.time())): val for t, val in zip(time_range, subset_data)}
        # print(json_data)
        
    print(json_data_tommo.keys)
    return json_data_today,json_data_tommo



# 최종 결과값:
  # final: BarrelTemperature_1~7을 하루 단위로 나눠 15분 간격으로 저장해둔 결과
    # 2023-07-07은 원본 온도 데이터, 2023-07-08은 예측된 온도 데이터
  # upper_bound_1~7: BarrelTemperature_1~7에 대한 온도 threshold, 이 온도를 넘기면 알람 발송
  # mean_temp: 2023-07-07 하루에 대한 BarrelTemperature_1~7의 평균 온도