import redis
import pandas as pd
import os
from datetime import date
from temperature_forecast import *
from send_redis import *

def getTemperature(redis_port=6379,redis_db=0):
    # Redis 연결 정보
    redis_host = "redis-my-redis-0d888-19180359-0042b61a74ce.kr.lb.naverncp.com"
    redis_pw=1213
    # Redis 클라이언트 생성
    redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db,password=redis_pw)
    # Get all keys in Redis
    today = str(date.today())
    print(today)
    all_keys = redis_client.keys(f"{today}*")
    result = {}

    for key in all_keys:
        data = redis_client.hgetall(key)
        timestamp = key.decode()
        result[timestamp] = {
            field.decode(): value.decode() for field, value in data.items()
        }

    return result


def predict_temperature():
    send_data()
    df = getTemperature()

    mold_drop = pd.DataFrame.from_dict(df,orient='index')
    mold_drop['TimeStamp']=mold_drop.index
    print(mold_drop)
    mold_drop[["Barrel_Temperature_1","Barrel_Temperature_2","Barrel_Temperature_3","Barrel_Temperature_4","Barrel_Temperature_5","Barrel_Temperature_6","Barrel_Temperature_7"]] = mold_drop[["Barrel_Temperature_1","Barrel_Temperature_2","Barrel_Temperature_3","Barrel_Temperature_4","Barrel_Temperature_5","Barrel_Temperature_6","Barrel_Temperature_7"]].apply(pd.to_numeric)
    mold_drop["TimeStamp"]=mold_drop["TimeStamp"].apply(pd.to_datetime)
    print(mold_drop.dtypes)
    # mold_drop.to_csv('molddrop.csv')
    
    Barrel_1 = pd.DataFrame({'ds': mold_drop['TimeStamp'], 'y': mold_drop['Barrel_Temperature_1']})
    Barrel_2 = pd.DataFrame({'ds': mold_drop['TimeStamp'], 'y': mold_drop['Barrel_Temperature_2']})
    Barrel_3 = pd.DataFrame({'ds': mold_drop['TimeStamp'], 'y': mold_drop['Barrel_Temperature_3']})
    Barrel_4 = pd.DataFrame({'ds': mold_drop['TimeStamp'], 'y': mold_drop['Barrel_Temperature_4']})
    Barrel_5 = pd.DataFrame({'ds': mold_drop['TimeStamp'], 'y': mold_drop['Barrel_Temperature_5']})
    Barrel_6 = pd.DataFrame({'ds': mold_drop['TimeStamp'], 'y': mold_drop['Barrel_Temperature_6']})
    Barrel_7 = pd.DataFrame({'ds': mold_drop['TimeStamp'], 'y': mold_drop['Barrel_Temperature_7']})
    
    Barrel_1 = timeseries(Barrel_1)
    Barrel_2 = timeseries(Barrel_2)
    Barrel_3 = timeseries(Barrel_3)
    Barrel_4 = timeseries(Barrel_4)
    Barrel_5 = timeseries(Barrel_5)
    Barrel_6 = timeseries(Barrel_6)
    Barrel_7 = timeseries(Barrel_7)

    

    temp_1 = Barrel_1
    temp_2 = Barrel_2
    temp_3 = Barrel_3
    temp_4 = Barrel_4
    temp_5 = Barrel_5
    temp_6 = Barrel_6
    temp_7 = Barrel_7
    
    # 각 데이터프레임의 평균 온도 구하기
    mean_temp_1 = temp_1['y'].mean()
    mean_temp_2 = temp_2['y'].mean()
    mean_temp_3 = temp_3['y'].mean()
    mean_temp_4 = temp_4['y'].mean()
    mean_temp_5 = temp_5['y'].mean()
    mean_temp_6 = temp_6['y'].mean()
    mean_temp_7 = temp_7['y'].mean()
    total_mean_temp = (mean_temp_1 + mean_temp_2 + mean_temp_3 + mean_temp_4 + mean_temp_5 + mean_temp_6 + mean_temp_7) / 7
    mean_temp = total_mean_temp.round()

    print(mean_temp)

    threshold = 1.5  # IQR에 대한 임계값 설정

    upper_bound_1 = calculate_upper_bound(temp_1['y'], threshold)
    upper_bound_2 = calculate_upper_bound(temp_2['y'], threshold)
    upper_bound_3 = calculate_upper_bound(temp_3['y'], threshold)
    upper_bound_4 = calculate_upper_bound(temp_4['y'], threshold)
    upper_bound_5 = calculate_upper_bound(temp_5['y'], threshold)
    upper_bound_6 = calculate_upper_bound(temp_6['y'], threshold)
    upper_bound_7 = calculate_upper_bound(temp_7['y'], threshold)

    print(upper_bound_1)
    print(upper_bound_2)
    print(upper_bound_3)
    print(upper_bound_4)
    print(upper_bound_5)
    print(upper_bound_6)
    print(upper_bound_7)

    model = Prophet()
    forecast_1 = data_forecast(temp_1,model)

    model_2 = Prophet()
    forecast_2 = data_forecast(temp_2,model_2)

    model_3 = Prophet()
    forecast_3 = data_forecast(temp_3,model_3)

    model_4 = Prophet()
    forecast_4 = data_forecast(temp_4,model_4)

    model_5 = Prophet()
    forecast_5 = data_forecast(temp_5,model_5)

    model_6 = Prophet()
    forecast_6 = data_forecast(temp_6,model_6)

    model_7 = Prophet()
    forecast_7 = data_forecast(temp_7,model_7)

    result_1 = inference(temp_1, forecast_1)
    result_2 = inference(temp_2, forecast_2)
    result_3 = inference(temp_3, forecast_3)
    result_4 = inference(temp_4, forecast_4)
    result_5 = inference(temp_5, forecast_5)
    result_6 = inference(temp_6, forecast_6)
    result_7 = inference(temp_7, forecast_7)

    final_forecast = pd.DataFrame()
    final_forecast['TimeStamp'] = result_1['TimeStamp']
    final_forecast['BarrelTemperature_1'] = result_1['Forecast']
    final_forecast['BarrelTemperature_2'] = result_2['Forecast']
    final_forecast['BarrelTemperature_3'] = result_3['Forecast']
    final_forecast['BarrelTemperature_4'] = result_4['Forecast']
    final_forecast['BarrelTemperature_5'] = result_5['Forecast']
    final_forecast['BarrelTemperature_6'] = result_6['Forecast']
    final_forecast['BarrelTemperature_7'] = result_7['Forecast']

    interval = 15
    final_today,final_tommo = split_data_by_day_and_interval(final_forecast, interval)

    # print(final)
    # print(final)
    
    final_df=pd.DataFrame.from_dict(final_today,orient='columns')
    today=date.today()
    # print(final_df)
    # print(final_df.columns)
    # print(final_df.index)
    redis_host = "redis-my-redis-0d888-19180359-0042b61a74ce.kr.lb.naverncp.com"
    redis_port = 6379
    redis_db = 1
    redis_pw = 1213
    redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db, password=redis_pw)

    # for t in final_df.index:
    #      for barrel in final_df.columns:
    #           print
    for t in final_df.index:
            redis_key = t+'_today'
            # print(redis_key,final_df[col][row])
            if redis_client.exists(redis_key):
                continue
            redis_client.hmset(
                redis_key, {
                     barrel: final_df[barrel][t]
                     for barrel in final_df.columns
                     }
            )
    final_df=pd.DataFrame.from_dict(final_tommo,orient='columns')

    for t in final_df.index:
            redis_key = t+'_tommo'
            # print(redis_key,final_df[col][row])
            if redis_client.exists(redis_key):
                continue
            redis_client.hmset(
                redis_key, {
                     barrel: final_df[barrel][t]
                     for barrel in final_df.columns
                     }
            )

    all_keys = redis_client.keys(f"*tommo")
    # print(all_keys)
    result = {}

    for key in all_keys:
        data = redis_client.hgetall(key)
        # print(data)
        timestamp = key.decode()
        result[timestamp] = {
            field.decode(): value.decode() for field, value in data.items()
        }
    print(result)
