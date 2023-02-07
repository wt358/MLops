
import os
from datetime import datetime

from kafka import KafkaConsumer
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from gridfs import GridFS
from numba import cuda
from json import loads

import numpy as np
import pandas as pd
import gridfs
import tensorflow as tf


from preprocess import *
from loadmodel import *
from logger import *
import params

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


    infer_lstm()
    print("hello inference the main products")

def infer_lstm():
    
    update_dt = datetime.now().strftime("%Y-%m-%d")
    test_data_path = params.test_data_path
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

        
    #data consumer
    
    now = datetime.now()
    curr_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    consumer = KafkaConsumer('test.coops2022_etl.etl_data',
            group_id=f'Inference_lstm_{curr_time}',
            bootstrap_servers=['kafka-clust-kafka-persis-d198b-11683092-d3d89e335b84.kr.lb.naverncp.com:9094'],
            value_deserializer=lambda x: loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            consumer_timeout_ms=10000
            )
    #consumer.poll(timeout_ms=1000, max_records=2000)

    #dataframe extract
    l=[]

    for message in consumer:
        message = message.value
        l.append(loads(message['payload'])['fullDocument'])
    df = pd.DataFrame(l)
    print(df)
    if df.empty:
        print("empty queue")
        return
    # dataframe transform
    df=df[df['idx']!='idx']
    print(df.shape)
    print(df.columns)
    print(df)

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
    db_model = client['coops2022_model']
    fs = gridfs.GridFS(db_model)
    collection_model=db_model['mongo_scaler_lstm']
    
    model_name = 'scaler_data'
    model_fpath = f'{model_name}.joblib'
    result = collection_model.find({"model_name": model_name}).sort('uploadDate', -1)
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
    collection_model=db_model[f'mongo_{model_name}']
    
    model_fpath = f'{model_name}.joblib'
    result = collection_model.find({"model_name": model_name}).sort('uploadDate', -1)
    
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
    scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
    scored['Threshold'] = 0.1
    scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    print(scored.head())

    y_test = scored['Anomaly']
    print(y_test.unique())


    print(y_test)

    db_test = client['coops2022_result']
    collection = db_test[f'result_{model_name}']
    #data=scored.to_dict('records')
    data=X_pred.to_dict('records')

    try:
        collection.insert_many(data,ordered=False)
    except Exception as e:
        print("mongo connection failer",e)



    print("hello inference lstm ae")
    
def infer_ocsvm():
    print("hello inference ocsvm")
    