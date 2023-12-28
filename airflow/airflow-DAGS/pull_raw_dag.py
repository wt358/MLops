from datetime import timedelta
from datetime import datetime
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models.variable import Variable
from airflow.utils.trigger_rule import TriggerRule

from sklearn.preprocessing import StandardScaler

import influxdb_client
import csv
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
import pandas as pd
import os

import time

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

from pattern_extract import *

molding_brand_name = ['WooJin', 'DongShin']
# molding_brand_name = ['WooJin' ]
woojin_factory_name = ['NewSeoGwang', ]
dongshin_factory_name = ['teng',  ]

def pull_influx_woojin(**kwargs):
    bucket = Variable.get("WOOJIN_INFLUX_BUCKET")
    org = Variable.get("WOOJIN_INFLUX_ORG")
    token = Variable.get("WOOJIN_INFLUX_TOKEN")
    # Store the URL of your InfluxDB instance
    url= Variable.get("WOOJIN_INFLUX_URL")
    start_date="-7d"

    print("WooJin")
   
    brand_name=kwargs['brand_name']
    factorys=eval(brand_name + "_factory_name")
    print(factorys)
    for factory in factorys:
        
        print(factory)
        arg_list=["CLSeoGwang06HO","CLSeoGwang25HO"]
        for arg in arg_list:
            client = influxdb_client.InfluxDBClient(
            url=url,
            token=token,
            org=org,
            timeout=500_000
            )
            query_api = client.query_api()

            #
            query = f' from(bucket:"{bucket}")\
            |> range(start: {start_date})\
            |> filter(fn:(r) => r._measurement == "{arg}")\
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")\
            '
            #|> range(start: -2mo)
            # result = query_api.query(org=org, query=query)
            # print(result)


            influx_df =  query_api.query_data_frame(query=query)
            print(influx_df.columns)
            if len(influx_df) < 1:
                client.close()
                return 0
            print(influx_df)
            client.close()
            data=influx_df.to_dict('records')
            host = Variable.get("WOOJIN_MONGO_URL_SECRET")
            client = MongoClient(host)

            db_test = client['peripheral_data']
            collection_test1 = db_test[f'{factory}_{arg}_peripheral_data']
            collection_test1.create_index([("_time",ASCENDING)],unique=True)
            try:
                result = collection_test1.insert_many(data,ordered=False)
            except Exception as e:
                print("mongo connection failed")
                print(e)
            client.close()
    print("hello pull influx")



# define funcs
# 이제 여기서 15분마다 실행되게 하고, query find 할때 20분 레인지
def pull_influx_dongshin(**kwargs):
    bucket = Variable.get("INFLUX_BUCKET")
    org = Variable.get("INFLUX_ORG")
    token = Variable.get("INFLUX_TOKEN")
    # Store the URL of your InfluxDB instance
    url= Variable.get("INFLUX_URL")
    start_date="-50d"
    print("Dongshin")
    brand_name=kwargs['brand_name']
    factorys=eval(brand_name + "_factory_name")
    print(factorys)
    
    for factory in factorys:
        client = influxdb_client.InfluxDBClient(
        url=url,
        token=token,
        org=org,
        timeout=500_000
        )

        query_api = client.query_api()

        #
        query = f' from(bucket:"{bucket}")\
        |> range(start: {start_date})\
        |> filter(fn:(r) => r._measurement == "NetworkInjectionMoldV1")\
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")\
        '
        #|> range(start: -2mo)
        # result = query_api.query(org=org, query=query)
        # print(result)

        influx_df =  query_api.query_data_frame(query=query)
        print(influx_df)
        print(influx_df.columns)
        if len(influx_df) < 1:
            client.close()
            return 0
        influx_df = influx_df[influx_df['All_Mold_Number']!="NaN"]
        client.close()
        data=influx_df.to_dict('records')
        host = Variable.get("MONGO_URL_SECRET")
        client = MongoClient(host)


        db_test = client['raw_data']
        collection_test1 = db_test[f'{factory}_mold_data']
        collection_test1.create_index([("_time",ASCENDING)],unique=True)
        try:
            # for row in data:
            #     uniq=row['_time']
            #     result = collection_test1.update_one({'idx':uniq},{"$set":row},upsert=True)
            result = collection_test1.insert_many(data,ordered=False)
        except Exception as e:
            print("mongo connection failed")
            print(e)
        client.close()
    print("hello pull influx")

def wait_kafka():
    time.sleep(30)
    
def pull_mssql_woojin(**kwargs):
    # DB 서버 주소
    print("hello mssql woojin")
    host = Variable.get("WOOJIN_MS_HOST")
    # 데이터 베이스 이름
    database = Variable.get("WOOJIN_MS_DATABASE")
    # 접속 유저명
    username = Variable.get("WOOJIN_MS_USERNAME")
    # 접속 유저 패스워드
    password = Variable.get("WOOJIN_MS_PASSWORD")
    #쿼리
    
    brand_name=kwargs['brand_name']
    factorys=eval(brand_name + "_factory_name")
    print(factorys)
    for factory in factorys:
        query = text("SELECT * from shot_data WITH(NOLOCK) where TimeStamp > DATEADD(DAY,-180,GETDATE())")
        # "SELECT * from shot_data WITH(NOLOCK) where TimeStamp > DATEADD(MONTH,-1,GETDATE())"
        #한시간 단위로 pull -> "SELECT *,DATEADD(MI,-60,GETDATE()) from shot_data WITH(NOLOCK)"
        # MSSQL 접속
        conection_url = sqlalchemy.engine.url.URL(
            drivername="mssql+pymssql",
            username=username,
            password=password,
            host=host,
            database=database,
        )
        engine = create_engine(conection_url,echo=True)

        #Session 생성
        #Session = sessionmaker(bind=engine)
        #ms_session = Session()

        FILENAME = "molding_all.csv"

        # sql_result = engine.execute(query)
        sql_result_pd = pd.read_sql_query(query, engine)
        df=sql_result_pd
        df=df[df['idx']!='idx']
        # print(sql_result)
        print(sql_result_pd)
        print(df)
        print(df.duplicated().sum())
        df1=df
        print(df1.drop_duplicates())
        
        # for i,row in enumerate(sql_result):
        #     #print(type(row)) == <class 'sqlalchemy.engine.row.LegacyRow'>
        #     for column, value in row.items():
        #         if i > 1:
        #             break
        #         print('{0}: {1}'.format(column, value))
        #         #outcsv.writerow(r)

        print("query result length: " + str(len(list(sql_result_pd))))
        data=df.to_dict('records')
        host_mongo = Variable.get("WOOJIN_MONGO_URL_SECRET")
        client = MongoClient(host_mongo)
        db_test = client['raw_data']
        collection_test1 = db_test[f'{factory}_mold_data']
        collection_test1.create_index([("TimeStamp",ASCENDING),
                                       ("Additional_Info_1",TEXT),
                                       ("Machine_Name",TEXT),
                                       ("Filling_Time",ASCENDING),
                                       ("Injection_Time",ASCENDING),
                                       ("Barrel_Temperature_1",ASCENDING),
                                       ("Max_Injection_Speed",ASCENDING),
                                       ("Cushion_Position",ASCENDING),
                                       ("Plasticizing_Time",ASCENDING),
                                       ]
                                      ,unique=True)
        try:
            result = collection_test1.insert_many(data,ordered=False)
        except Exception as e:
            print("mongo connection failed",e)
        client.close()
        engine.dispose()
    
def pull_transform_dongshin(**kwargs):

    brand_name=kwargs['brand_name']
    factorys=eval(brand_name + "_factory_name")
    print(factorys)
    for factory in factorys:
        host = Variable.get("MONGO_URL_SECRET")
        client = MongoClient(host)
        db_test = client['raw_data']
        collection_test1 = db_test[f'{factory}_mold_data']
        now = datetime.now()
        start = now - timedelta(days=50)
        print(start)
        query={
                '_time':{
                    '$gt':start,
                    '$lt':now
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
        df.drop(columns={'_id','_time','result','_measurement','table','_start','_stop'},inplace=True)

        df=df.drop_duplicates(subset=["idx"])
        df.rename(columns={"idx":"_time"},inplace=True)
        for i in df.columns:
            print("컬럼: {:40s}, 크기: {}, Null: {}".format(i, df[i].shape, df[i].isnull().sum()))
        
        print(df.shape)
        print(df.columns)
        print(df)
        df.loc[:,df.columns.drop('_time')]=df.loc[:,df.columns.drop('_time')].apply(pd.to_numeric,errors='coerce')
        print(df)
        print(df.info())
        # tmp = df.corr().abs()
        # print(tmp)
        # asdf = []
        # for i in df.columns[1:]:
        #     corr_col = tmp[tmp[i] > 0.5][i]
        #     if len(corr_col.index.tolist()) > 0: 
        #         asdf.append(corr_col.drop(i))
        
        # highly_correlated_columns = np.unique([j for i in asdf for j in i.index])
        # print(df[highly_correlated_columns].corr())
        
        # scaler = StandardScaler()
        # df2 = pd.DataFrame(scaler.fit_transform(df[highly_correlated_columns].dropna()), columns = highly_correlated_columns)
        
        # temp = pd.DataFrame([[i, df2[i].value_counts().shape[0]] for i in highly_correlated_columns if df2[i].value_counts().shape[0]]) 
        # PV_Hold_Press_First_Time까지 cut (통계 기반 threshold를 구하는건 의미 없음)
        # print(temp.sort_values(by=1, ascending=False))
        
        # important_column = temp[temp[1] >= 40][0].tolist()
        # print(important_column)
        
        # for i in important_column:
        #     try:
        #         gwt = get_work_time(df[i], 0.25, 10, 10)
        #         plt.figure(figsize=(16, 6))
        #         df[i].plot(figsize=(15,5), title=i)
        #         pd.concat([df[i].iloc[j[0]:j[1]] for j in gwt]).plot()
        #     except:
        #         print(i)
        
        # important_column2 = []
        # gwts = []
        # for i in important_column:
        #     try:
        #         gwt = get_work_time(df[i], 0.25, 10, 10)
        #         gwts.append(gwt)
        #         important_column2.append(i)
        #     except:
        #         print(i)
        # print(important_column2)
        
        # print(df)
        important_column2=['All_Mold_Number',
            'Injection_Time',
            'Machine_Process_Time',
            'PV_Cooling_Time',
            'PV_Penalty_Neglect_Monitoring',
            'Product_Process_Time',
            'Reservation_Mold_Number',
            'Screw_Position',
            'Weighing_Speed']
        df=df[important_column2+['_time']].dropna()
        print(df)
        host = Variable.get("MONGO_URL_SECRET")
        client = MongoClient(host)
        today=datetime.now().strftime("%Y-%m-%d")
        db_test = client['etl_data']
        collection_etl=db_test[f'etl_{factory}']
        collection_etl.create_index([("_time",ASCENDING)],unique=True)
        data=df.to_dict('records')
        # 아래 부분은 테스트 할 때 매번 다른 oid로 데이터가 쌓이는 것을 막기 위함
        try:
            result = collection_aug.insert_many(data,ordered=False)
        except Exception as e: 
            print("mongo connection failed")
            print(e)
        collection_aug=db_test[f'test_{factory}']
        collection_aug.create_index([("_time",ASCENDING)],unique=True)
        try:
            result = collection_aug.insert_many(data,ordered=False)
        except Exception as e: 
            print("mongo connection failed")
            print(e)
        client.close()
    print("hello")

def pull_transform_woojin(**kwargs):
    
    brand_name=kwargs['brand_name']
    factorys=eval(brand_name + "_factory_name")
    print(factorys)
    for factory in factorys:
        print(factory)
        host = Variable.get("WOOJIN_MONGO_URL_SECRET")
        client = MongoClient(host)

        db_test = client['raw_data']
        collection_test1 = db_test[f'{factory}_mold_data']
        now = datetime.now()
        start = now - timedelta(days=180)
        print(start)
        query={
                'TimeStamp':{
                    '$gt':start,
                    '$lt':now
                    }
                }
        try:
            df = pd.DataFrame(list(collection_test1.find(query)))
        except Exception as e:
            print(e) 
            print("mongo connection failed")
        
        if df.empty:
            print("empty")
            return
        print(df)
        df.drop(columns={'_id'},inplace=True)
        client.close()
        # df=df.drop_duplicates(subset=["idx"])
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
        '''
        moldset_labeled_9000R=df[df.Additional_Info_1=='09520 9000R']
        print(moldset_labeled_9000R.head())
        moldset_labeled_9000R=moldset_labeled_9000R.reset_index(drop=True)
        print(moldset_labeled_9000R.head())
        print(len(moldset_labeled_9000R))
        '''
        host = Variable.get("WOOJIN_MONGO_URL_SECRET")
        client = MongoClient(host)

        db_test = client['etl_data']
        collection_etl=db_test[f'etl_{factory}']
        collection_etl.create_index([("TimeStamp",ASCENDING),
                                     ("Additional_Info_1",TEXT),
                                       ("Machine_Name",TEXT),
                                       ("Filling_Time",ASCENDING),
                                       ("Injection_Time",ASCENDING),
                                       ("Barrel_Temperature_1",ASCENDING),
                                       ("Max_Injection_Speed",ASCENDING),
                                       ("Cushion_Position",ASCENDING),
                                       ("Plasticizing_Time",ASCENDING),
                                     ],unique=True)
        data=df.to_dict('records')
        try:
            result = collection_etl.insert_many(data,ordered=False)
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
    for i in molding_brand_name:
        i=i.lower()
        fact=f'{i}_factory_name'
        fact_list=eval(fact)
        dummy1 = DummyOperator(task_id="path1"+i)
        sleep_task = PythonOperator(
            task_id="sleep_60s"+ i,
            python_callable=wait_kafka,
            # depends_on_past=True,
            depends_on_past=False,
            owner="coops2",
            retries=3,
            retry_delay=timedelta(minutes=1),
        )

        t1 = PythonOperator(
            task_id="pull_influx"+ i ,
            python_callable=eval("pull_influx_"+ i),
            op_kwargs={'brand_name':i},
            # depends_on_past=True,
            depends_on_past=False,
            owner="coops2",
            retries=3,
            retry_delay=timedelta(minutes=1),
        )

        t3 = PythonOperator(
            task_id="pull_transform"+ i,
            python_callable=eval("pull_transform_"+ i),
            op_kwargs={'brand_name':i},
            depends_on_past=False,
            owner="coops2",
            retries=3,
            retry_delay=timedelta(minutes=1),
        )
        
        dummy2 = DummyOperator(task_id="path2"+ i,trigger_rule=TriggerRule.NONE_FAILED)
        if i == "woojin":
            t2 = PythonOperator(
               task_id="pull_mssql"+i,
                python_callable=eval("pull_mssql_"+ i),
                op_kwargs={'brand_name':i},
                depends_on_past=False,
                owner="coops2",
                retries=3,
                retry_delay=timedelta(minutes=1), 
            )
            # dummy1 >> [t1,t2] >> dummy2
            dummy1 >> t2 >> dummy2
        else:
            dummy1 >> t1>> dummy2
            

        
# 테스크 순서를 정합니다.
# t1 실행 후 t2를 실행합니다.


        dummy2 >> t3 >> sleep_task 