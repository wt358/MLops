from datetime import timedelta
from datetime import datetime
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models.variable import Variable
from airflow.utils.trigger_rule import TriggerRule


import influxdb_client
import csv
from pymongo import MongoClient
import pandas as pd
import os

import time

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

# define funcs
# 이제 여기서 15분마다 실행되게 하고, query find 할때 20분 레인지
def pull_influx():
    bucket = Variable.get("INFLUX_BUCKET")
    org = Variable.get("INFLUX_ORG")
    token = Variable.get("INFLUX_TOKEN")
    # Store the URL of your InfluxDB instance
    url= Variable.get("INFLUX_URL")


    client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org
    )

    query_api = client.query_api()

    #
    query = ' from(bucket:"cloud-bucket")\
    |> range(start: -24h)\
    |> filter(fn:(r) => r._measurement == "NetworkInjectionMoldV1")\
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")\
    '
    #|> range(start: -2mo)
    # result = query_api.query(org=org, query=query)
    # print(result)


    FILENAME = 'Weight.csv'

    influx_df =  query_api.query_data_frame(query=query)
    print(len(influx_df))
    client.close()
    data=influx_df.to_dict('records')
    host = Variable.get("MONGO_URL_SECRET")
    client = MongoClient(host)


    db_test = client['coops2022']
    collection_test1 = db_test['weight_data']
    try:
        result = collection_test1.insert_many(data)
    except:
        print("mongo connection failed")
    client.close()

def wait_kafka():
    time.sleep(30)

def pull_transform():
    host = Variable.get("MONGO_URL_SECRET")
    client = MongoClient(host)

    db_test = client['coops2022']
    collection_test1 = db_test['molding_data']
    now = datetime.now()
    start = now - timedelta(days=3)
    print(start)
    query={
            'TimeStamp':{
                '$gt':f'{start}',
                '$lt':f'{now}'
                }
            }
    try:
        df = pd.DataFrame(list(collection_test1.find(query)))
    except Exception as e: 
        print("mongo connection failed")
    
    print(df)
    if df.empty:
        print("empty")
        return
    df.drop(columns={'_id'},inplace=True)

    df=df.drop_duplicates(subset=["idx"])
    df.drop(columns={'Mold_Temperature_1',
        'Mold_Temperature_2',
        'Mold_Temperature_3',
        'Mold_Temperature_4',
        'Mold_Temperature_5',
        'Mold_Temperature_6',
        'Mold_Temperature_7',
        'Mold_Temperature_8',
        'Mold_Temperature_9',
        'Mold_Temperature_10',
        'Mold_Temperature_11',
        'Mold_Temperature_12',
        'Hopper_Temperature',
        'Cavity',
        'NGmark',
        },inplace=True)
    df=df[df['idx']!='idx']
    print(df.shape)
    print(df.columns)
    print(df)
    
    host = Variable.get("MONGO_URL_SECRET")
    client = MongoClient(host)

    db_test = client['coops2022_etl']
    collection_aug=db_test['etl_data']
    data=df.to_dict('records')
    # 아래 부분은 테스트 할 때 매번 다른 oid로 데이터가 쌓이는 것을 막기 위함
    try:
        for row in data:
            uniq=row['idx']
            result = collection_aug.update_one({'idx':uniq},{"$set":row},upsert=True)
    except Exception as e: 
        print("mongo connection failed")
        print(e)
    client.close()
    print("hello")


# define DAG with 'with' phase
with DAG(
    dag_id="pull_raw_dag", # DAG의 식별자용 아이디입니다.
    description="pull raw data from local DBs", # DAG에 대해 설명합니다.
    start_date=days_ago(2), # DAG 정의 기준 2일 전부터 시작합니다.
    schedule_interval=timedelta(days=1), # 매일 00:00에 실행합니다.
    tags=["my_dags"],
    max_active_runs=3,
    ) as dag:
# define the tasks

#t = BashOperator(
#    task_id="print_hello",
#    bash_command="echo Hello",
#    owner="", # 이 작업의 오너입니다. 보통 작업을 담당하는 사람 이름을 넣습니다.
#    retries=3, # 이 테스크가 실패한 경우, 3번 재시도 합니다.
#    retry_delay=timedelta(minutes=5), # 재시도하는 시간 간격은 5분입니다.
#)

    sleep_task = PythonOperator(
        task_id="sleep_60s",
        python_callable=wait_kafka,
        depends_on_past=True,
        owner="coops2",
        retries=0,
        retry_delay=timedelta(minutes=1),
    )

    t1 = PythonOperator(
        task_id="pull_influx",
        python_callable=pull_influx,
        depends_on_past=True,
        owner="coops2",
        retries=0,
        retry_delay=timedelta(minutes=1),
    )

    
    t3 = PythonOperator(
        task_id="pull_transform",
        python_callable=pull_transform,
        depends_on_past=True,
        owner="coops2",
        retries=0,
        retry_delay=timedelta(minutes=1),
    )
   

    dummy1 = DummyOperator(task_id="path1")
    dummy2 = DummyOperator(task_id="path2",trigger_rule=TriggerRule.NONE_FAILED)
    
    # 테스크 순서를 정합니다.
    # t1 실행 후 t2를 실행합니다.
    
    dummy1 >> t1>> dummy2

    dummy2 >> t3 >> sleep_task 