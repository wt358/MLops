
import os
from datetime import datetime
from datetime import timedelta 

from kafka import KafkaConsumer
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pymongo
from gridfs import GridFS
from numba import cuda
from json import loads

import numpy as np
import pandas as pd
import gridfs
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, Dataset

from preprocess import *
from loadmodel import *
from logger import *
from predict_temp import *
from sklearn.svm import OneClassSVM 
import params
from knowledge_distillation import *

def infer_tad():
    logging.info('########## START INFERENCE ##########')
    logging.info('- GPU envrionmental')

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:  
        # kill all the processes which is using GPU
        device = cuda.get_current_device()
        device.reset()

        # Set GPU env: prevent GPU from out-of-memory
        # ref: https://www.tensorflow.org/guide/gpu?hl=ko
        try: 
            for gpu in gpus: 
                tf.config.experimental.set_memory_growth(gpu, True) 
        except RuntimeError as e: 
            logging.info(e)
        logging.info(gpus)
    else:
        logging.info('NO CPU')


    ####################################################################################
    logging.info('########## START INFERENCE ##########')
    logging.info('- Root directory: {}'.format(os.path.abspath(os.curdir)))
    logging.info('Parameters')


    print("hello inference tad")
    
def infer_ocsvm():
    #data consumer
    now = datetime.now()
    curr_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    print(os.environ['EXECUTION_DATE'])
    # consumer = KafkaConsumer('test.coops2022_etl.etl_data',
    #         group_id=f'Inference_vari_model_{curr_time}',
    #         bootstrap_servers=['kafka-clust-kafka-persis-d198b-11683092-d3d89e335b84.kr.lb.naverncp.com:9094'],
    #         value_deserializer=lambda x: loads(x.decode('utf-8')),
    #         auto_offset_reset='earliest',
    #         consumer_timeout_ms=10000
    #         )
    #consumer.poll(timeout_ms=1000, max_records=2000)

    #dataframe extract
    # l=[]

    # for message in consumer:
    #     message = message.value
    #     l.append(loads(message['payload'])['fullDocument'])
    # df = pd.DataFrame(l)
    # consumer.close()
    factory=os.environ['FACT_NAME']
    host = os.environ['MONGO_URL_SECRET'] 
    client = MongoClient(host)
    print(factory)
    db_test = client['etl_data']
    collection_etl=db_test[f'etl_{factory}']
    start=now-timedelta(days=4)
    query={
            'TimeStamp':{
                '$gt':start,
                '$lt':now
                }
            }
    try:
        df = pd.DataFrame(list(collection_etl.find(query)))
    except Exception as e:
        print("mongo connection failed", e)
    client.close()
    print(df)
    if df.empty:
        print("empty queue")
        return
    # dataframe transform
    df=df[df['idx']!='idx']
    df['TimeStamp']=pd.to_datetime(df['TimeStamp'],utc=True)
    now=now.astimezone()
    print(now)
    start_time=now-timedelta(minutes=30)
    start_time=now-timedelta(days=4)
    print(start_time)
    df=df[df['TimeStamp']>=start_time]
    print(df.shape)
    print(df.columns)
    print(df)

    df.drop(columns={'_id',
        },inplace=True)
    
    print(df)
    if df.empty:
        print("empty")
        return 1
    
    labled = pd.DataFrame(df, columns = ['Filling_Time','Plasticizing_Time','Cycle_Time','Cushion_Position'])


    labled.columns = map(str.lower,labled.columns)
    labled.rename(columns={'class':'label'},inplace=True)
    print(labled.head())

    target_columns = pd.DataFrame(labled, columns = ['cycle_time', 'cushion_position'])
    target_columns.astype('float')
     
     
    host = os.environ['MONGO_URL_SECRET'] 
    client=MongoClient(host)
    db_model = client['model_var']
    fs = gridfs.GridFS(db_model)
    collection_model=db_model[f'OCSVM_{factory}']
    
    model_name = 'OC_SVM'
    model_fpath = f'{model_name}.joblib'
    result = collection_model.find({"model_name": model_name}).sort([("inserted_time", -1)])
    print(result)
    # cnt=len(list(result.clone()))
    # print(result[0])
    # print(result[cnt-1])
    try:
        file_id = str(result[0]['file_id'])
        model = LoadModel(mongo_id=file_id).clf
    except Exception as e:
        print("exception occured in oc_svm",e)
        model = OneClassSVM(kernel = 'rbf', gamma = 0.001, nu = 0.04).fit(target_columns)
    joblib.dump(model, model_fpath)
    
    print(model.get_params())
    
    y_pred = model.predict(target_columns)
    print(y_pred)



    # filter outlier index
    outlier_index = np.where(y_pred == -1)

    #filter outlier values
    outlier_values = target_columns.iloc[outlier_index]
    print(outlier_values)
    
    # 이상값은 -1으로 나타낸다.
    score = model.fit(target_columns)
    anomaly = model.predict(target_columns)
    target_columns['anomaly']= anomaly
    anomaly_data = target_columns.loc[target_columns['anomaly']==-1] 
    print(target_columns['anomaly'].value_counts())

    target_columns[target_columns['anomaly']==1] = 0
    target_columns[target_columns['anomaly']==-1] = 1
    target_columns['Anomaly'] = target_columns['anomaly'] > 0.5
    y_test = target_columns['Anomaly']
    
    print(y_test.unique())

    # df = pd.DataFrame(labled, columns = ['label'])
    # print(df.label)
    
    # outliers = df['label']
    # outliers = outliers.fillna(0)
    # print(outliers.unique())
    # print(outliers)

    print(y_test)
    y_log=pd.DataFrame(index=df.index)
    y_log['TimeStamp']=df['TimeStamp']
    y_log['Anomaly']=y_test
    print(y_log['TimeStamp'].nunique())
    # outliers = outliers.to_numpy()
    # y_test = y_test.to_numpy()

    # get (mis)classification
    # cf = confusion_matrix(outliers, y_test)

    # # true/false positives/negatives
    # print(cf)
    # (tn, fp, fn, tp) = cf.flatten()

    # print(f"""{cf}
    # % of transactions labeled as fraud that were correct (precision): {tp}/({fp}+{tp}) = {tp/(fp+tp):.2%}
    # % of fraudulent transactions were caught succesfully (recall):    {tp}/({fn}+{tp}) = {tp/(fn+tp):.2%}
    # % of g-mean value : root of (specificity)*(recall) = ({tn}/({fp}+{tn})*{tp}/({fn}+{tp})) = {(tn/(fp+tn)*tp/(fn+tp))**0.5 :.2%}""")
    db_test = client['result_log']
    collection = db_test[f'log_{model_name}_{factory}']
    collection.create_index([("TimeStamp",pymongo.ASCENDING)],unique=True)
    data=y_log.to_dict('records')
    try:
        collection.insert_many(data,ordered=False)
    except Exception as e:
        print("mongo connection failed",e)
    client.close()
    print("hello oc_svm inference")

def infer_student():
    #data consumer
    now = datetime.now()
    curr_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    print(os.environ['EXECUTION_DATE'])
    # consumer = KafkaConsumer('test.coops2022_etl.etl_data',
    #         group_id=f'Inference_vari_model_{curr_time}',
    #         bootstrap_servers=['kafka-clust-kafka-persis-d198b-11683092-d3d89e335b84.kr.lb.naverncp.com:9094'],
    #         value_deserializer=lambda x: loads(x.decode('utf-8')),
    #         auto_offset_reset='earliest',
    #         consumer_timeout_ms=10000
    #         )
    #consumer.poll(timeout_ms=1000, max_records=2000)

    #dataframe extract
    # l=[]

    # for message in consumer:
    #     message = message.value
    #     l.append(loads(message['payload'])['fullDocument'])
    # df = pd.DataFrame(l)
    # consumer.close()
    factory=os.environ['FACT_NAME']
    host = os.environ['MONGO_URL_SECRET'] 
    client = MongoClient(host)
    print(factory)
    db_test = client['raw_data']
    collection_etl=db_test[f'{factory}_mold_data']
    start=now-timedelta(days=4)
    query={
            'TimeStamp':{
                '$gt':start,
                '$lt':now
                }
            }
    try:
        df = pd.DataFrame(list(collection_etl.find(query)))
    except Exception as e:
        print("mongo connection failed", e)
    client.close()
    print(df)
    if df.empty:
        print("empty queue")
        return
    # dataframe transform
    df=df[df['idx']!='idx']
    df['TimeStamp']=pd.to_datetime(df['TimeStamp'],utc=True)
    now=now.astimezone()
    print(now)
    start_time=now-timedelta(minutes=30)
    start_time=now-timedelta(days=4)
    print(start_time)
    df=df[df['TimeStamp']>=start_time]
    print(df.shape)
    print(df.columns)
    print(df)

    
    print(df)
    if df.empty:
        print("empty")
        return 1
    host = os.environ['MONGO_URL_SECRET'] 
    client=MongoClient(host)
    db_model = client['model_var']
    fs = gridfs.GridFS(db_model)
    collection_model=db_model[f'student_{factory}']
    
    model_name = 'student'
    model_fpath = f'{model_name}.joblib'
    result = collection_model.find({"model_name": model_name}).sort([("inserted_time", -1)])
    print(result)
    # cnt=len(list(result.clone()))
    # print(result[0])
    # print(result[cnt-1])
    try:
        file_id = str(result[0]['file_id'])
        student_model = LoadModel(mongo_id=file_id).clf
    except Exception as e:
        print("exception occured in student",e)
        # model = OneClassSVM(kernel = 'rbf', gamma = 0.001, nu = 0.04).fit(target_columns)
    joblib.dump(student_model, model_fpath)
    
    collection_model=db_model[f'teacher_{factory}']
    
    model_name = 'teacher'
    model_fpath = f'{model_name}.joblib'
    result = collection_model.find({"model_name": model_name}).sort([("inserted_time", -1)])
    print(result)
    # cnt=len(list(result.clone()))
    # print(result[0])
    # print(result[cnt-1])
    try:
        file_id = str(result[0]['file_id'])
        teacher_model = LoadModel(mongo_id=file_id).clf
    except Exception as e:
        print("exception occured in teacher",e)
        # model = OneClassSVM(kernel = 'rbf', gamma = 0.001, nu = 0.04).fit(target_columns)
    joblib.dump(teacher_model, model_fpath)
    # print(model.get_params())
    
    # info_9000a = df[(df['Additional_Info_1']=='9000a 09520')]
    info_9000a = df
    # info_9000a = info_9000a.reset_index(drop=True)
    datasets = info_9000a

    test_datasets, test_time = preprocess(datasets)
    final_datasets = torch.tensor(test_datasets.values)

    test_loader = DataLoader(final_datasets, shuffle= False)
    
    print(test_loader)
    print(test_time)
    start_time = time.time()

    result = inference(student_model, teacher_model, test_loader,test_time)

    end_time = time.time()

    elapsed_time = end_time - start_time # 모델 학습 시간
    result['time'] = elapsed_time
    print(f"학습 시간: {elapsed_time}초")
    print(result)
    db_result=client['result_log']
    collection_result = db_result['log_kd_NewSeoGwang']
    collection_result.create_index([("TimeStamp",pymongo.ASCENDING)],unique=True)
    data=result.to_dict('records')
    try:
        collection_result.insert_many(data,ordered=False)
    except Exception as e:
        print("Exception occured",e)


    
    client.close()
    print("hello oc_svm inference")



def infer_main():
    logging.info('########## START INFERENCE ##########')
    logging.info('- GPU envrionmental')

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:  
        # kill all the processes which is using GPU
        device = cuda.get_current_device()
        device.reset()

        # Set GPU env: prevent GPU from out-of-memory
        # ref: https://www.tensorflow.org/guide/gpu?hl=ko
        try: 
            for gpu in gpus: 
                tf.config.experimental.set_memory_growth(gpu, True) 
        except RuntimeError as e: 
            logging.info(e)
        logging.info(gpus)
    else:
        logging.info('NO CPU')


    ####################################################################################
    logging.info('########## START INFERENCE ##########')
    logging.info('- Root directory: {}'.format(os.path.abspath(os.curdir)))
    logging.info('Parameters')

    predict_temperature()
    infer_lstm()
    
    print("hello inference the main products")

def infer_lstm():
    
    update_dt = datetime.now().strftime("%Y-%m-%d")
    test_data_path = params.train_data_path
    train_columns = params.train_columns
    time_columns = params.time_columns
    interval =params.interval
    latent_dim = params.latent_dim
    shape = params.shape
    encoder_input_shape = params.encoder_input_shape
    generator_input_shape = params.generator_input_shape
    critic_x_input_shape = params.critic_x_input_shape
    critic_z_input_shape = params.critic_z_input_shape
    encoder_reshape_shape = params.encoder_reshape_shape
    generator_reshape_shape = params.generator_reshape_shape
    learning_rate = params.learning_rate
    batch_size = params.batch_size
    n_critics = params.n_critics
    epochs = params.epochs
    check_point = params.check_point
    z_range =params.z_range
    window_size = params.window_size
    window_size_portion = params.window_size_portion
    window_step_size = params.window_step_size
    window_step_size_portion =params.window_step_size_portion
    min_percent = params.min_percent
    anomaly_padding =params.anomaly_padding

    factory=os.environ['FACT_NAME']
    host = os.environ['MONGO_URL_SECRET'] 
    client = MongoClient(host)
    print(factory)
    db_test = client['etl_data']
    collection_etl=db_test[f'etl_{factory}']
   
    #data consumer
    
    now = datetime.now()
    curr_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    # consumer = KafkaConsumer('test.coops2022_etl.etl_data',
    #         group_id=f'Inference_lstm_{curr_time}',
    #         bootstrap_servers=['kafka-clust-kafka-persis-d198b-11683092-d3d89e335b84.kr.lb.naverncp.com:9094'],
    #         value_deserializer=lambda x: loads(x.decode('utf-8')),
    #         auto_offset_reset='earliest',
    #         consumer_timeout_ms=10000
    #         )
    #consumer.poll(timeout_ms=1000, max_records=2000)

    #dataframe extract
    # l=[]

    # for message in consumer:
    #     message = message.value
    #     l.append(loads(message['payload'])['fullDocument'])
    # df = pd.DataFrame(l)
    start=now-timedelta(days=4)
    # start1=now-timedelta(days=num)
    query={
            'TimeStamp':{
                '$gt':start,
                # '$lt':start1
                '$lt': now
                }
            }
    try:
        df = pd.DataFrame(list(collection_etl.find(query)))
    except Exception as e:
        print("mongo connection failed", e)

    print(df)
    if df.empty:
        print("empty queue")
        return
    # dataframe transform
    df=df[df['idx']!='idx']
    df['TimeStamp']=pd.to_datetime(df['TimeStamp'],utc=True)
    now=now.astimezone()
    print(now)
    start_time=(now-timedelta(minutes=30)).astimezone()
    start_time=(now-timedelta(days=4)).astimezone()
    print(start_time)
    df=df[df['TimeStamp']>=start_time]
    print(df.shape)
    print(df.columns)
    print(df)
    if df.empty:
        print("empty df")
        return 1

    df.drop(columns={'_id',
        },inplace=True)
    
    print(df)

    
    labled = pd.DataFrame(df, columns = ['Filling_Time','Plasticizing_Time','Cycle_Time','Cushion_Position'])


    labled.columns = map(str.lower,labled.columns)

    print(labled.head())

    X_test = labled.sample(frac=1)
    test = X_test 
    X_test=X_test.values
    print(f"""Shape of the datasets:
        Testing  (rows, cols) = {X_test.shape}""")
    
    host = os.environ['MONGO_URL_SECRET'] 
    client = MongoClient(host)
    db_model = client['model_var']
    fs = gridfs.GridFS(db_model)
    collection_model=db_model[f'scaler_lstm_{factory}']
    
    model_name = 'scaler_data'
    model_fpath = f'{model_name}.joblib'
    result = collection_model.find({"model_name": model_name}).sort([("inserted_time", -1)])
    print(result)
    if len(list(result.clone()))==0:
        print("empty")
        scaler = MinMaxScaler()
    else:
        print("not empty")
        file_id = str(result[0]['file_id'])
        scaler = LoadModel(mongo_id=file_id).clf
    
    # transforming data from the time domain to the frequency domain using fast Fourier transform
    test_fft = np.fft.fft(X_test)
    
    X_test = scaler.transform(X_test)# 나중에 scaler도 pull raw 2 aug에서 모델을 저장해서 놓고 여기서는 그 모델을 불러와서 transform(X_test)만 해야함.
    scaler_filename = "scaler_data"
    joblib.dump(scaler, scaler_filename)

    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    print("Test data shape:", X_test.shape)
    #model load    
    
    model_name = 'LSTM_autoencoder'
    collection_model=db_model[f'{model_name}_{factory}']
    
    model_fpath = f'{model_name}.joblib'
    result = collection_model.find({"model_name": model_name}).sort([("inserted_time", -1)])
    
    print(result)
    if len(list(result.clone()))==0:
        print("empty")
        return 1
    else:
        print("not empty")
        file_id = str(result[0]['file_id'])
        model = LoadModel(mongo_id=file_id).clf

    joblib.dump(model, model_fpath)
    
    model.compile(optimizer='adam', loss='mae')
    
    # 이상값은 -1으로 나타낸다.
    print(model.summary())


    X_pred = model.predict(X_test)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = pd.DataFrame(X_pred, columns=test.columns)
    X_pred.index = test.index
    
    scored = pd.DataFrame(index=test.index)
    Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
    scored['TimeStamp']=pd.to_datetime(df['TimeStamp'])
    scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
    loss_list=np.abs(X_pred-Xtest)
    print(loss_list)
    mean=np.mean(loss_list,axis=0)
    std=np.cov(loss_list.T)
    print(mean)
    print(std)
    x=loss_list-mean
    print(x)

    new_df=np.mean(np.abs(np.dot(np.dot(x,std),x.T)),axis=1).reshape(-1,1)
    scaler_minmax=StandardScaler()
    scaler_minmax.fit(new_df)
    data_scaler1=np.abs(scaler_minmax.transform(new_df))
    scaler_minmax=MinMaxScaler()
    scaler_minmax.fit(new_df)
    data_scaler2=scaler_minmax.transform(new_df)

    # print(data_scaler)
    scored['Anomaly_Score_standard']=data_scaler1
    scored['Anomaly_Score_minmax']=data_scaler2
    scored['Threshold'] = 0.1
    scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    print(scored.head())

    y_test = scored['Anomaly']
    print(y_test.unique())


    print(y_test)
    client=MongoClient(host)
    db_test = client['result_log']
    collection = db_test[f'log_{model_name}_{factory}']
    collection.create_index([("TimeStamp",pymongo.ASCENDING)],unique=True)
    data=scored.to_dict('records')
    # data=X_pred.to_dict('records')

    try:
        collection.insert_many(data,ordered=False)
    except Exception as e:
        print("mongo connection failer",e)
    client.close()
    print("hello inference lstm ae")
    
    