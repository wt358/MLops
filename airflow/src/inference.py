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
from pymongo import ASCENDING, DESCENDING

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
        logging.info('NO GPU')


    ####################################################################################
    logging.info('########## START INFERENCE ##########')
    logging.info('- Root directory: {}'.format(os.path.abspath(os.curdir)))
    logging.info('Parameters')


    print("hello inference tad")

def infer_main():
    logging.info('########## START INFERENCE ##########')
    
    # logging.info('- CPU envrionmental')
    # logging.info('OC_SVM in CPU environment')

    # infer_ocsvm()
    
    logging.info('- GPU environmental')

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
        logging.info('NO GPU')


    ####################################################################################
    logging.info('########## START INFERENCE ##########')
    logging.info('- Root directory: {}'.format(os.path.abspath(os.curdir)))
    logging.info('Parameters')


    infer_lstm()
    print("hello inference the main products")
    
def infer_ocsvm():
    
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
    consumer = KafkaConsumer('etl.etl_data.test_teng',
        group_id=f'infer_ocsvm_{curr_time}',
        bootstrap_servers=['kafka-clust-kafka-persis-cc65d-15588214-38845b0307b9.kr.lb.naverncp.com:9094'],
        value_deserializer=lambda x: loads(x.decode('utf-8')),
        auto_offset_reset='earliest',
        consumer_timeout_ms=10000
        )
    
    #consumer.poll(timeout_ms=1000, max_records=2000)

    #dataframe extract
    l=[]

    for message in consumer:
        message = message.value
        try:
            l.append(loads(message['payload'])['fullDocument'])
        except:
            print(message)

    df = pd.DataFrame(l)
    consumer.close()
    print(df)
    if df.empty:
        print("empty queue")
        return
    # dataframe transform
    print(df.shape)
    print(df.columns)
    print(df)

    df.drop(columns={'_id',
        },inplace=True)
    
    df.rename(columns={'_time':'TimeStamp'},inplace=True)
    df['TimeStamp']=df['TimeStamp'].apply(lambda x : x['$date'])
    df['TimeStamp']=df['TimeStamp'].apply(lambda x : datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
    print(df)

    important_columns = ['All_Mold_Number','Injection_Time' ,'Machine_Process_Time'  ,'PV_Cooling_Time', 'PV_Penalty_Neglect_Monitoring' ,'Product_Process_Time'  ,'Reservation_Mold_Number'  ,'Screw_Position'  ,'Weighing_Speed','Class']
    
    labled = pd.DataFrame(df, columns = important_columns)
    labled.columns = map(str.lower,labled.columns)
    # labled.rename(columns={'class':'label'},inplace=True)
    print(labled.head())
    important_columns.remove('Class')
    target_columns = pd.DataFrame(labled, columns = map(str.lower,important_columns))

    target_columns.astype('float')
    
    host = os.environ['MONGO_URL_SECRET'] 
    client = MongoClient(host)
     
    db_model = client['model_var']
    fs = gridfs.GridFS(db_model)
    collection_model=db_model['OCSVM_teng']
    
    model_name = 'OC_SVM'
    model_fpath = f'{model_name}.joblib'
    result = collection_model.find({"model_name": model_name}).sort([("inserted_time", -1)])
    print(result)
    cnt=len(list(result.clone()))
    print(cnt)
    
    try:
        file_id = str(result[0]['file_id'])
        model = LoadModel(mongo_id=file_id).clf
    except Exception as e:
        print("exception occured in oc_svm",e)
        return 1
    joblib.dump(model, model_fpath)
    
    print(model.get_params())
    # print(model.summary())
    
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

    # df1 = pd.DataFrame(labled, columns = ['label'])
    # print(df1.label)
    
    # outliers = df1['label']
    # outliers = outliers.fillna(0)
    # print(outliers.unique())
    # print(outliers)
    
    print(y_test)
    y_log = pd.DataFrame(index=df.index)
    y_log['TimeStamp']=pd.to_datetime(df['TimeStamp'])
    y_log['Anomaly']=y_test
    # outliers = outliers.to_numpy()
    # y_test = y_test.to_numpy()

    # get (mis)classification
    # cf = confusion_matrix(outliers, y_test)

    # true/false positives/negatives
    # print(cf)
    # (tn, fp, fn, tp) = cf.flatten()
    db_test = client['result_log']
    collection = db_test[f'log_{model_name}_teng']
    collection.create_index([("TimeStamp",ASCENDING)],unique=True)
    #data=scored.to_dict('records')
    data=y_log.to_dict('records')

    try:
        collection.insert_many(data,ordered=False)
    except Exception as e:
        print("mongo connection failer",e)


    # print(f"""{cf}
    # % of transactions labeled as fraud that were correct (precision): {tp}/({fp}+{tp}) = {tp/(fp+tp):.2%}
    # % of fraudulent transactions were caught succesfully (recall):    {tp}/({fn}+{tp}) = {tp/(fn+tp):.2%}
    # % of g-mean value : root of (specificity)*(recall) = ({tn}/({fp}+{tn})*{tp}/({fn}+{tp})) = {(tn/(fp+tn)*tp/(fn+tp))**0.5 :.2%}""")
 
 
    client.close()
    print("hello infer ocsvm")
    
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
    consumer = KafkaConsumer('etl.etl_data.test_teng',
        group_id=f'infer_lstm_{curr_time}',
        bootstrap_servers=['kafka-clust-kafka-persis-cc65d-15588214-38845b0307b9.kr.lb.naverncp.com:9094'],
        value_deserializer=lambda x: loads(x.decode('utf-8')),
        auto_offset_reset='earliest',
        consumer_timeout_ms=10000
        )
    
    #consumer.poll(timeout_ms=1000, max_records=2000)

    #dataframe extract
    l=[]

    for message in consumer:
        message = message.value
        try:
            l.append(loads(message['payload'])['fullDocument'])
        except:
            print(message)
    df = pd.DataFrame(l)
    consumer.close()
    print(df)
    if df.empty:
        print("empty queue")
        return
    # dataframe transform
    print(df.shape)
    print(df.columns)
    print(df)

    df.drop(columns={'_id',
        },inplace=True)
    
    df.rename(columns={'_time':'TimeStamp'},inplace=True)
    df['TimeStamp']=df['TimeStamp'].apply(lambda x : x['$date'])
    df['TimeStamp']=df['TimeStamp'].apply(lambda x : datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
    print(df)

    important_columns = ['All_Mold_Number','Injection_Time' ,'Machine_Process_Time'  ,'PV_Cooling_Time', 'PV_Penalty_Neglect_Monitoring' ,'Product_Process_Time'  ,'Reservation_Mold_Number'  ,'Screw_Position'  ,'Weighing_Speed','Class']
    important_columns.remove('Class')
    labled = pd.DataFrame(df, columns = important_columns)

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
    collection_model=db_model['scaler_lstm_teng']
    
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
    collection_model=db_model[f'{model_name}_teng']
    
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
    scored['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)

    loss_list=np.abs(X_pred-Xtest).T
    print(loss_list)
    mean=np.mean(loss_list,axis=0)
    std=np.cov(loss_list.T)
    print(mean)
    print(std)
    x=loss_list-mean
    print(x)
    # print(np.matmul(x,std))
    print(np.mean(np.matmul(np.matmul(x,std),x.T),axis=1))
    scored['Anomaly_Score']=np.mean(np.matmul(np.matmul(x,std),x.T),axis=1)
    scored['Threshold'] = 0.1
    scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    print(scored.head())

    y_test = scored['Anomaly']
    print(y_test.unique())


    print(y_test)

    db_test = client['result_log']
    collection = db_test[f'log_{model_name}_teng']
    collection.create_index([("TimeStamp",ASCENDING)],unique=True)
    #data=scored.to_dict('records')
    data=scored.to_dict('records')

    try:
        collection.insert_many(data,ordered=False)
    except Exception as e:
        print("mongo connection failer",e)



    print("hello inference lstm ae")
    
    