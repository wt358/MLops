from datetime import timedelta
from datetime import datetime

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score, classification_report,  confusion_matrix

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM 
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from numba import cuda
from functools import partial
from scipy import integrate, stats



from gan import buildGAN
from preprocess import *
from logger import *
from cTadGAN import *
from loadmodel import *
from data_evaluation import *
from inference import *
import params


# from IPython.display import Image
# import matplotlib.pyplot as plt
# import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns
from bson import ObjectId

import gridfs
import io

from gridfs import GridFS



from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from tensorflow.python.client import device_lib
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import  Model,Sequential

import joblib

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import csv
import pandas as pd
import os
import sys
import math

import time
import numpy as np

from collections import Counter


from kafka import KafkaConsumer
from kafka import KafkaProducer

from pymongo import MongoClient

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

import json
from json import loads
import random as rn


np.random.seed(34)

# manual parameters
RANDOM_SEED = int(os.environ['RANDOM_SEED'])
TRAINING_SAMPLE = int(os.environ['TRAINING_SAMPLE'])
VALIDATE_SIZE = float(os.environ['VALIDATE_SIZE'])

# setting random seeds for libraries to ensure reproducibility
np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

rn.seed(10)

tf.random.set_seed(10)

# define funcs
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model


#pull raw data in the cloud and run the aug module. Then save the aug data files in the local.

def iqr_mds_gan():
    now = datetime.now()
    curr_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    consumer = KafkaConsumer('test.coops2022_etl.etl_data',
            group_id=f'airflow_{curr_time}',
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
        try:
            l.append(loads(message['payload'])['fullDocument'])
        except:
            print(message)
    df = pd.DataFrame(l)
    print(df)
    consumer.close()
    # dataframe transform
    df=df[df['idx']!='idx']
    print(df.shape)
    print(df.columns)
    print(df)

    df=df[df['Additional_Info_1'].str.contains("9000a")]
    df.drop(columns={'_id',
        },inplace=True)

    df=df[['idx', 'Machine_Name','Additional_Info_1', 'Additional_Info_2','Shot_Number','TimeStamp',
            'Average_Back_Pressure', 'Average_Screw_RPM', 'Clamp_Close_Time',
            'Clamp_Open_Position', 'Cushion_Position', 'Cycle_Time', 'Filling_Time',
            'Injection_Time', 'Plasticizing_Position',
            'Plasticizing_Time', 'Switch_Over_Position',
            ]]
    #IQR
    print(df)
    print(df.dtypes)
    
    df=df.reset_index(drop=True)
    section=df
    section=IQR(section)
    print(section)

    # data frame 자르기
    last_idx = 0
    curr_idx = 0

    # 자른 데이터프레임을 저장할 리스트
    pds = []
    section=section.reset_index(drop=True)
    print(section.index.tolist())
    for idx in range(1,len(section.index.tolist())):
        # print(moldset_labeled_9000R.loc[idx,'TimeStamp'])
        time_to_compare1 = datetime.strptime(section.loc[idx,'TimeStamp'], "%Y-%m-%d %H:%M:%S")
        time_to_compare2 = datetime.strptime(section.loc[idx-1,'TimeStamp'], "%Y-%m-%d %H:%M:%S")
        time_diff = time_to_compare1 - time_to_compare2

        # 분 단위로 비교
        if time_diff.seconds / 60 > 15:
            curr_idx = idx-1
            pds.append(section.truncate(before=last_idx, after=curr_idx,axis=0))
            last_idx = idx

    else:
        pds.append(section.truncate(before=last_idx, after=len(section.index.tolist())-1,axis=0))

    for i in range(len(pds)):
        print(i, pds[i].count().max())

    print(pds[0])
    df_all=MDS_molding(pds)

    print(df_all)
    print(df_all.columns)

    #GAN

    df=df_all
    df['Class'] = df_all['Class'].map(lambda x: 1 if x == -1 else 0)

    print(df)
    print(df['Class'].value_counts(normalize=True)*100)

    print(f"Number of Null values: {df.isnull().any().sum()}")

    print(f"Dataset has {df.duplicated().sum()} duplicate rows")

    df=df.dropna()
    df.drop_duplicates(inplace=True)
    try:
        df.drop(columns={'Labeling'}
                ,inplace=True)
    except:
        print("passed")
    

    print(df)

    # checking skewness of other columns

    print(df.drop('Class',1).skew())
    
    # skew_cols = df.iloc[:,5:].drop('Class',1).skew().loc[lambda x: x>2].index
    # print(skew_cols)

    # print(device_lib.list_local_devices())
    # print(tf.config.list_physical_devices())
    
    with tf.device("/gpu:0"):
    #     for col in skew_cols:
    #         lower_lim = abs(df[col].min())
    #         normal_col = df[col].apply(lambda x: np.log10(x+lower_lim+1))
    #         print(f"Skew value of {col} after log transform: {normal_col.skew()}")
    
    #     scaler = StandardScaler()
    #     #scaler = MinMaxScaler()
    #     X = scaler.fit_transform(df.iloc[:,5:].drop('Class', 1))
    #     y = df['Class'].values
    #     print(X.shape, y.shape)

    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)


    #     gan = buildGAN(out_shape=X_train.shape[1], num_classes=2)
    #     # cgan.out_shape=X_train.shape[1]

    #     y_train = y_train.reshape(-1,1)
    #     pos_index = np.where(y_train==1)[0]
    #     neg_index = np.where(y_train==0)[0]
    #     gan.train(X_train, y_train, pos_index, neg_index, epochs=50)#원래 epochs= 5000

    #     print(df.shape)
    #     print(X_train.shape)
    #     gan_num=df.shape[0]
    #     noise = np.random.normal(0, 1, (gan_num, 32))
    #     sampled_labels = np.ones(gan_num).reshape(-1, 1)

    #     gen_samples = gan.generator.predict([noise, sampled_labels])
    #     gen_samples = scaler.inverse_transform(gen_samples)
    #     print(gen_samples.shape)

    #     gen_df = pd.DataFrame(data = gen_samples,
    #             columns = df.iloc[:,5:].drop('Class',1).columns)
    #     gen_df['Class'] = 1
    #     print(gen_df)

    #     Class0 = df[df['Class'] == 0 ]
    #     print(Class0)

    #     pca = PCA(n_components = 2)
    #     PC = pca.fit_transform(gen_df)
    #     PCdf = pca.fit_transform(Class0.iloc[:,5:])

    #     VarRatio = pca.explained_variance_ratio_
    #     VarRatio = pd.DataFrame(np.round_(VarRatio,3))

    #     CumVarRatio    = np.cumsum(pca.explained_variance_ratio_)
    #     CumVarRatio_df = pd.DataFrame(np.round_(CumVarRatio,3))

    #     Result = pd.concat([VarRatio , CumVarRatio_df], axis=1)
    #     print(pd.DataFrame(Result))
    #     print(pd.DataFrame(PC))

    #     pca3 = PCA(n_components = 3)
    #     PC3 = pca3.fit_transform(gen_df)
    #     PC_df = pca3.fit_transform(Class0.iloc[:,5:])

    #     VarRatio3 = pca3.explained_variance_ratio_
    #     VarRatio3 = pd.DataFrame(np.round_(VarRatio3,3))

    #     CumVarRatio3    = np.cumsum(pca3.explained_variance_ratio_)
    #     CumVarRatio_df3 = pd.DataFrame(np.round_(CumVarRatio3,3))

    #     Result3 = pd.concat([VarRatio3 , CumVarRatio_df3], axis=1)
    #     print(pd.DataFrame(Result3))
    #     print(pd.DataFrame(PC3))

    #     augdata = pd.concat([pd.DataFrame(Class0), gen_df])
    #     Augdata = augdata.reset_index(drop=True)
    #     print(Augdata)
    #     print(Augdata['Class'].value_counts(normalize=True)*100)
    #     Augdata['TimeStamp']=pd.to_datetime(Augdata['TimeStamp'],unit='s')
        Augdata=df
        mongoClient = MongoClient()
        #host = Variable.get("MONGO_URL_SECRET")
        host = os.environ['MONGO_URL_SECRET'] 
        client = MongoClient(host)

        db_test = client['coops2022_aug']
        collection_aug=db_test['mongo_aug1']
        data=Augdata.to_dict('records')
    # 아래 부분은 테스트 할 때 매번 다른 oid로 데이터가 쌓이는 것을 막기 위함
        try:
            isData = collection_aug.find_one()
            if len(isData) !=0:
                print("collection is not empty")
                collection_aug.delete_many({})
            try:
                result = collection_aug.insert_many(data,ordered=False)
            except Exception as e:
                print("mongo connection failed", e)
        except:
            print("there is no collection")
            try:
                result = collection_aug.insert_many(data,ordered=False)
            except Exception as e:
                print("mongo connection failed", e)
        client.close()
    print("hello")


#provide the aug data that saved in the local to the aug topic in the kafka cluster
def oc_svm():
    
    mongoClient = MongoClient()
    #host = Variable.get("MONGO_URL_SECRET")
    host = os.environ['MONGO_URL_SECRET'] 
    client = MongoClient(host)
    
    db_test = client['coops2022_aug']
    collection_aug=db_test['mongo_aug1']
    
    try:
        moldset_df = pd.DataFrame(list(collection_aug.find()))
        
    except:
        print("mongo connection failed")
        return False
    
    print(moldset_df)

    moldset_9000R=moldset_df
    

    labled = pd.DataFrame(moldset_9000R, columns = ['Filling_Time','Plasticizing_Time','Cycle_Time','Cushion_Position','Class'])
    labled.columns = map(str.lower,labled.columns)
    labled.rename(columns={'class':'label'},inplace=True)
    print(labled.head())

    target_columns = pd.DataFrame(labled, columns = ['cycle_time', 'cushion_position'])
    target_columns.astype('float')
     
    db_model = client['coops2022_model']
    fs = gridfs.GridFS(db_model)
    collection_model=db_model['mongo_OCSVM']
    
    model_name = 'OC_SVM'
    model_fpath = f'{model_name}.joblib'
    result = collection_model.find({"model_name": model_name}).sort([("inserted_time", -1)])
    print(result)
    cnt=len(list(result.clone()))
    print(result[0])
    print(result[cnt-1])
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

    df = pd.DataFrame(labled, columns = ['label'])
    print(df.label)
    
    outliers = df['label']
    outliers = outliers.fillna(0)
    print(outliers.unique())
    print(outliers)

    print(y_test)

    outliers = outliers.to_numpy()
    y_test = y_test.to_numpy()

    # get (mis)classification
    cf = confusion_matrix(outliers, y_test)

    # true/false positives/negatives
    print(cf)
    (tn, fp, fn, tp) = cf.flatten()

    print(f"""{cf}
    % of transactions labeled as fraud that were correct (precision): {tp}/({fp}+{tp}) = {tp/(fp+tp):.2%}
    % of fraudulent transactions were caught succesfully (recall):    {tp}/({fn}+{tp}) = {tp/(fn+tp):.2%}
    % of g-mean value : root of (specificity)*(recall) = ({tn}/({fp}+{tn})*{tp}/({fn}+{tp})) = {(tn/(fp+tn)*tp/(fn+tp))**0.5 :.2%}""")
    

    
    #save model in the DB
    # save the local file to mongodb
    with open(model_fpath, 'rb') as infile:
        file_id = fs.put(
                infile.read(), 
                model_name=model_name
                )
    # insert the model status info to ModelStatus collection 
    params = {
            'model_name': model_name,
            'file_id': file_id,
            'inserted_time': datetime.now()
            }
    result = collection_model.insert_one(params)
    client.close()

    print("hello OC_SVM")

def lstm_autoencoder():
    mongoClient = MongoClient()
    #host = Variable.get("MONGO_URL_SECRET")
    host = os.environ['MONGO_URL_SECRET'] 
    client = MongoClient(host)
    
    db_test = client['coops2022_aug']
    collection_aug=db_test['mongo_aug1']
    
    try:
        moldset_df = pd.DataFrame(list(collection_aug.find()))
        
    except:
        print("mongo connection failed")
        return False
    
    print(moldset_df)

    outlier = moldset_df[moldset_df.Class == 1]
    print(outlier.head())
    labled = pd.DataFrame(moldset_df, columns = ['Filling_Time','Plasticizing_Time','Cycle_Time','Cushion_Position','Class'])

    labled.columns = map(str.lower,labled.columns)
    labled.rename(columns={'class':'label'},inplace=True)
    print(labled.head()) 
    
    

    # splitting by class
    fraud = labled[labled.label == 1]
    clean = labled[labled.label == 0]

    print(f"""Shape of the datasets:
        clean (rows, cols) = {clean.shape}
        fraud (rows, cols) = {fraud.shape}""")
    
    # shuffle our training set
    clean = clean.sample(frac=1).reset_index(drop=True)

    # training set: exlusively non-fraud transactions
    global TRAINING_SAMPLE 
    if clean.shape[0] < TRAINING_SAMPLE:
        TRAINING_SAMPLE=(clean.shape[0]//5)*4

    X_train = clean.iloc[:TRAINING_SAMPLE].drop('label', axis=1)
    train = clean.iloc[:TRAINING_SAMPLE].drop('label', axis=1)

    # testing  set: the remaining non-fraud + all the fraud 
    X_test = clean.iloc[TRAINING_SAMPLE:].append(fraud).sample(frac=1)
    test = clean.iloc[TRAINING_SAMPLE:].append(fraud).sample(frac=1)
    test.drop('label', axis = 1, inplace = True)
    # 여기 test set이랑 train set 겹침

    print(f"""Our testing set is composed as follows:

            {X_test.label.value_counts()}""")
    
    X_test, y_test = X_test.drop('label', axis=1).values, X_test.label.values

    print(f"""Shape of the datasets:
        training (rows, cols) = {X_train.shape}
        Testing  (rows, cols) = {X_test.shape}""")

    with tf.device("/gpu:0"):

        # transforming data from the time domain to the frequency domain using fast Fourier transform
        train_fft = np.fft.fft(X_train)
        test_fft = np.fft.fft(X_test)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        scaler_filename = "scaler_data"

        # reshape inputs for LSTM [samples, timesteps, features]
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        print("Training data shape:", X_train.shape)
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        print("Test data shape:", X_test.shape)
       
        #scaler and lstm autoencoder model save
        db_model = client['coops2022_model']
        fs = gridfs.GridFS(db_model)
        collection_model=db_model['mongo_scaler_lstm']
       
        model_fpath = f'{scaler_filename}.joblib'
        joblib.dump(scaler, model_fpath)
        
        # save the local file to mongodb
        with open(model_fpath, 'rb') as infile:
            file_id = fs.put(
                    infile.read(), 
                    model_name=scaler_filename
                    )
        # insert the model status info to ModelStatus collection 
        params = {
                'model_name': scaler_filename,
                'file_id': file_id,
                'inserted_time': datetime.now()
                }
        result = collection_model.insert_one(params)


        # load the model
        collection_model=db_model['mongo_LSTM_autoencoder']
        
        model_name = 'LSTM_autoencoder'
        model_fpath = f'{model_name}.joblib'
        result = collection_model.find({"model_name": model_name}).sort([("inserted_time", -1)])
        print(result)
        cnt=len(list(result.clone()))
        print(cnt)
        print(result[0])
        print(result[cnt-1])
        try:
            file_id = str(result[0]['file_id'])
            model = LoadModel(mongo_id=file_id).clf
        except Exception as e:
            print("exception occured in lstm ae",e)
            model = autoencoder_model(X_train)
        
        joblib.dump(model, model_fpath)
        
        model.compile(optimizer='adam', loss='mae')
        
        # 이상값은 -1으로 나타낸다.
        print(model.summary())

        nb_epochs = 100
        batch_size = 10
        history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size, validation_split=0.05).history

        X_pred = model.predict(X_train)
        X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
        X_pred = pd.DataFrame(X_pred, columns=train.columns)
        X_pred.index = train.index

        scored = pd.DataFrame(index=train.index)
        Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
        scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)

        plt.figure(figsize=(16,9), dpi=80)
        plt.title('Loss Distribution', fontsize=16)
        sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
        plt.xlim([0.0,.5])
        plt.show()


        # calculate the loss on the test set
        X_pred = model.predict(X_test)
        X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
        X_pred = pd.DataFrame(X_pred, columns=test.columns)
        X_pred.index = test.index

        scored = pd.DataFrame(index=test.index)
        Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
        scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
        scored['Threshold'] = 0.1
        scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
        scored['label'] = labled['label']
        print(scored.head())


        y_test = scored['Anomaly']
        print(y_test.unique())

        print(scored[scored['Anomaly']==True].label.count())
        print(scored.label.unique())

        outliers = scored['label']
        outliers = outliers.fillna(0)
        print(outliers.unique())

        outliers = outliers.to_numpy()
        y_test = y_test.to_numpy()
        print(y_test)
        cm = confusion_matrix(y_test, outliers)
        (tn, fp, fn, tp) = cm.flatten()
        
        print(f"""{cm}
        % of transactions labeled as fraud that were correct (precision): {tp}/({fp}+{tp}) = {tp/(fp+tp):.2%}
        % of fraudulent transactions were caught succesfully (recall):    {tp}/({fn}+{tp}) = {tp/(fn+tp):.2%}
        % of g-mean value : root of (specificity)*(recall) = ({tn}/({fp}+{tn})*{tp}/({fn}+{tp})) = {(tn/(fp+tn)*tp/(fn+tp))**0.5 :.2%}""")

        print(roc_auc_score(outliers, y_test))
    
    
        db_model = client['coops2022_model']
        fs = gridfs.GridFS(db_model)
        collection_model=db_model['mongo_LSTM_autoencoder']
       
        model_name = 'LSTM_autoencoder'
        model_fpath = f'{model_name}.joblib'
        joblib.dump(model, model_fpath)
        
        # save the local file to mongodb
        with open(model_fpath, 'rb') as infile:
            file_id = fs.put(
                    infile.read(), 
                    model_name=model_name
                    )
        # insert the model status info to ModelStatus collection 
        params = {
                'model_name': model_name,
                'file_id': file_id,
                'inserted_time': datetime.now()
                }
        result = collection_model.insert_one(params)
    client.close()

    print("hello auto encoder")

def teng():
    
    logging.info('########## START UPDATE ##########')
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
    logging.info('- Parameters')

    
    now = datetime.now()
    train_dt = now.strftime("%Y-%m-%d_%H:%M:%S")

    train_data_path = params.train_data_path
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
    usecols=params.usecols


    logging.info('Step 1. Preprocess data')
    # Read Data and reconstruct data fromat for TadGAN

    consumer = KafkaConsumer('test.coops2022_etl.etl_data',
            group_id=f'teng_{train_dt}',
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
        try:
            l.append(loads(message['payload'])['fullDocument'])
        except:
            print(message)
    df = pd.DataFrame(l)
    print(df)
    consumer.close()
    # dataframe transform
    df=df[df['idx']!='idx']
    print(df.shape)
    print(df.columns)
    print(df)

    df.drop(columns={'_id',
        },inplace=True)
    df=df[['idx', 'Machine_Name','Additional_Info_1', 'Additional_Info_2','Shot_Number','TimeStamp',
            'Average_Back_Pressure', 'Average_Screw_RPM', 'Clamp_Close_Time',
            'Clamp_Open_Position', 'Clamp_open_time','Cushion_Position', 'Cycle_Time', 'Filling_Time',
            'Injection_Time', 'Plasticizing_Position','Max_Back_Pressure','Max_Injection_Pressure','Max_Injection_Speed','Max_Screw_RPM',
            'Max_Switch_Over_Pressure',
            'Plasticizing_Time', 'Switch_Over_Position','Barrel_Temperature_1','Barrel_Temperature_2','Barrel_Temperature_3',
            'Barrel_Temperature_4','Barrel_Temperature_5','Barrel_Temperature_6','Barrel_Temperature_7',
            ]]


    
    df=df[df['Machine_Name'] != '7']
    df=df[df['Machine_Name'] != '6i']
    df=df[df['Machine_Name'] != '']
    #EDA
    print(df)
    
    df=df.drop(columns=['idx','Machine_Name','Additional_Info_1','Additional_Info_2','Shot_Number'],axis=1)
    df=df.dropna(axis=0)
    print(df)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    print("to number")
    print(df.head().T)
    print(df.info())

    print(df.describe())
    for i in df.columns:
        print("컬럼: {:40s}, 크기: {}, Null: {}".format(i, df[i].shape, df[i].isnull().sum()))
    print(df)
    
    scaler = MinMaxScaler()
    q = [0.025, 0.975]
    for i in df.columns[1:]:
        lb, ub = df[i].quantile(q).tolist()
        samp = df[i].map(lambda x: None if (x < lb)or(x > ub) else x)
        samp = scaler.fit_transform(samp.to_frame())
        fig,ax = plt.subplots(1,1)
        pd.Series(samp.reshape(-1,)).plot(figsize=(30,5), title=i, ax=ax)
    print(df)


    df_scaled,scaled=outlier_iqr(df)
    
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    scaled_imp = imp_mean.fit_transform(scaled)

    df_scaled = pd.DataFrame(scaled_imp).T
    df_scaled.columns = df.columns[1:]
    df_scaled.index = pd.to_datetime(df['TimeStamp'])
    
    print(df_scaled.info())
    #초단위 groupby
    target = []
    for u in usecols:
        target_i = df_scaled[u]
        target_i = target_i.reset_index().groupby('TimeStamp').mean()
        target.append(target_i)
    df_target = pd.concat(target,axis=1)
    print(df_target)
    df = df_target[['Plasticizing_Time', 'Max_Switch_Over_Pressure', 'Cycle_Time', 'Max_Injection_Pressure', 'Barrel_Temperature_6']]
    print(df)
    os.makedirs('./data/cleansed',exist_ok=True)
    df.to_csv(train_data_path,encoding='utf-8-sig')
    
    train_dataset = data_reshape(train_data_path,vib_columns = train_columns)
    train_dt="today"
    
    for train_column in train_columns:
        train_data = train_dataset[train_column]

        # check train data size
        if train_data.shape[0] <= 100:
            logging.info('Data for train/inference should be larger than rolling size (100)')
            break

        # Check Intervals
        interval_unique = np.unique([train_data['timestamp'][i+1] 
                                     - train_data['timestamp'][i] for i in range(train_data.shape[0]-1)])
        if len(interval_unique) != 1:
            print('Time Intervals between data points are not equal. All the intervals are changed to one')
            index_origin = train_data['timestamp']
            train_data.loc[:,'timestamp'] = range(train_data.shape[0])
            interval = train_data['timestamp'][1] - train_data['timestamp'][0] 
        else:
            interval = interval_unique[0]
            index_origin = None

        # TimeSegments 
        X, index = time_segments_aggregate(train_data, interval=interval, time_column='timestamp')

        # Imputer
        imp = SimpleImputer()
        X = imp.fit_transform(X)

        # MinMax
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X)

        # Create rolling window sequences
        X, _, _, _ = rolling_window_sequences(X, 
                                              index, 
                                              window_size=100, 
                                              target_size=1, 
                                              step_size=1,
                                              target_column=0)
        os.makedirs('./data/preprocessed/{}'.format(train_dt), exist_ok=True)
        np.save('./data/preprocessed/{}/{}.npy'.format(train_dt, train_column), X)
        np.save('./data/preprocessed/{}/{}_index.npy'.format(train_dt, train_column), index)
        np.save('./data/preprocessed/{}/{}_index_origin.npy'.format(train_dt, train_column), index_origin)
        logging.info("Training data input shape: {} - {}".format(train_column, X.shape))


    for train_column in train_columns:
        ####################################################################################
        logging.info('Step 2. Build TadGAN network')

        # Layer Parameters
        encoder = build_encoder_layer(input_shape=encoder_input_shape,
                                      encoder_reshape_shape=encoder_reshape_shape)
        generator = build_generator_layer(input_shape=generator_input_shape,
                                          generator_reshape_shape=generator_reshape_shape)
        critic_x = build_critic_x_layer(input_shape=critic_x_input_shape)
        critic_z = build_critic_z_layer(input_shape=critic_z_input_shape)

        optimizer = tf.keras.optimizers.Adam(learning_rate)

        # Critic x
        z = Input(shape=(latent_dim, 1))
        x = Input(shape=shape)
        x_ = generator(z)
        z_ = encoder(x)
        fake_x = critic_x(x_)
        valid_x = critic_x(x)
        interpolated_x = RandomWeightedAverage()([x, x_])
        critic_x_model = Model(inputs=[x, z], outputs=[valid_x, fake_x, interpolated_x])

        # Critic z
        fake_z = critic_z(z_)
        valid_z = critic_z(z)
        interpolated_z = RandomWeightedAverage()([z, z_])
        critic_z_model = Model(inputs=[x, z], outputs=[valid_z, fake_z, interpolated_z])

        # Encoder Decoder
        z_gen = Input(shape=(latent_dim, 1))
        x_gen_ = generator(z_gen)
        x_gen = Input(shape=shape)
        z_gen_ = encoder(x_gen)
        x_gen_rec = generator(z_gen_)
        fake_gen_x = critic_x(x_gen_)
        fake_gen_z = critic_z(z_gen_)
        encoder_generator_model = Model([x_gen, z_gen], [fake_gen_x, fake_gen_z, x_gen_rec])
        
        ####################################################################################
        logging.info('Step 3. Train the network & Save models')
        
        # Load data for training
        X_path = './data/preprocessed/{}/{}.npy'.format(train_dt, train_column)
        index_path = './data/preprocessed/{}/{}_index.npy'.format(train_dt, train_column)

        X = np.load(X_path, allow_pickle=True)
        index = np.load(index_path, allow_pickle=True)
        print(X)
        print(index)

        # Load model for training
        encoder_model_path = './model/model_{}/{}/encoder'.format(train_dt, train_column)
        generator_model_path = './model/model_{}/{}/decoder'.format(train_dt, train_column)
        critic_x_model_path = './model/model_{}/{}/critic_x'.format(train_dt, train_column)   
        critic_z_model_path = './model/model_{}/{}/critic_z'.format(train_dt, train_column)  

        critic_x_m_model_path = './model/model_{}/{}/critic_x_model'.format(train_dt, train_column)
        critic_z_m_model_path = './model/model_{}/{}/critic_z_model'.format(train_dt, train_column)
        encoder_generator_model_path = './model/model_{}/{}/encoder_generator_model'.format(train_dt, train_column)

        # Log path for tensorboard 
        train_log_dir = 'tensorlog/' + train_dt + '/' + train_column
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # define loss for tensorboard
        tb_epoch_cx_loss = tf.keras.metrics.Mean('epoch_cx_loss', dtype=tf.float32)
        tb_epoch_cz_loss = tf.keras.metrics.Mean('epoch_cz_loss', dtype=tf.float32)
        tb_epoch_eg_loss = tf.keras.metrics.Mean('epoch_eg_loss', dtype=tf.float32)
        tb_loss = tf.keras.metrics.Sum('epoch_loss', dtype=tf.float32)

        # functions for training
        @tf.function
        def critic_x_train_on_batch(x, z, valid, fake, delta):
            with tf.GradientTape() as tape:

                (valid_x, fake_x, interpolated) = critic_x_model(inputs=[x, z], training=True) 

                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    pred = critic_x(interpolated[0], training=True)

                grads = gp_tape.gradient(pred, interpolated)[0]
                grads = tf.square(grads)
                ddx = tf.sqrt(1e-8 + tf.reduce_sum(grads, axis=np.arange(1, len(grads.shape))))
                gp_loss = tf.reduce_mean((ddx - 1.0) ** 2)

                loss = tf.reduce_mean(wasserstein_loss(valid, valid_x))
                loss += tf.reduce_mean(wasserstein_loss(fake, fake_x))
                loss += gp_loss*10.0

            gradients = tape.gradient(loss, critic_x_model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, critic_x_model.trainable_weights))

            tb_epoch_cx_loss(loss)
            return loss

        @tf.function
        def critic_z_train_on_batch(x, z, valid, fake, delta):
            with tf.GradientTape() as tape:

                (valid_z, fake_z, interpolated) = critic_z_model(inputs=[x, z], training=True)

                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    pred = critic_z(interpolated[0], training=True)

                grads = gp_tape.gradient(pred, interpolated)[0]
                grads = tf.square(grads)
                ddx = tf.sqrt(1e-8 + tf.reduce_sum(grads, axis=np.arange(1, len(grads.shape))))
                gp_loss = tf.reduce_mean((ddx - 1.0) ** 2)

                loss = tf.reduce_mean(wasserstein_loss(valid, valid_z))
                loss += tf.reduce_mean(wasserstein_loss(fake, fake_z))
                loss += gp_loss*10.0        

            gradients = tape.gradient(loss, critic_z_model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, critic_z_model.trainable_weights))

            tb_epoch_cz_loss(loss)
            return loss

        @tf.function
        def enc_gen_train_on_batch(x, z, valid):
            with tf.GradientTape() as tape:

                (fake_gen_x, fake_gen_z, x_gen_rec) = encoder_generator_model(inputs=[x, z], training=True)

                x = tf.squeeze(x)
                x_gen_rec = tf.squeeze(x_gen_rec)

                loss = tf.reduce_mean(wasserstein_loss(valid, fake_gen_x))
                loss += tf.reduce_mean(wasserstein_loss(valid, fake_gen_z))
                loss += tf.keras.losses.MSE(x, x_gen_rec)*10
                loss = tf.reduce_mean(loss)

            gradients = tape.gradient(loss, encoder_generator_model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, encoder_generator_model.trainable_weights))

            tb_epoch_eg_loss(loss)
            return loss

        # Train 
        X = X.reshape((-1, shape[0], 1))
        X_ = np.copy(X)

        fake = np.ones((batch_size, 1), dtype=np.float32)
        valid = -np.ones((batch_size, 1), dtype=np.float32)
        delta = np.ones((batch_size, 1), dtype=np.float32)

        
        train_loss = []
        for epoch in range(1, epochs+1):

            np.random.shuffle(X_)

            epoch_eg_loss = []
            epoch_cx_loss = []
            epoch_cz_loss = []

            minibatches_size = batch_size * n_critics
            num_minibatches = int(X_.shape[0] // minibatches_size)

            for i in range(num_minibatches):
                minibatch = X_[i * minibatches_size: (i + 1) * minibatches_size]

                generator.trainable = False
                encoder.trainable = False

                # train critics 
                for j in range(n_critics):
                    x = minibatch[j * batch_size: (j + 1) * batch_size]
                    z = np.random.normal(size=(batch_size, latent_dim, 1))
                    epoch_cx_loss.append(critic_x_train_on_batch(x, z, valid, fake, delta))
                    epoch_cz_loss.append(critic_z_train_on_batch(x, z, valid, fake, delta))

                critic_x.trainable = False
                critic_z.trainable = False        
                generator.trainable = True
                encoder.trainable = True     

                # train encoder, generator   
                epoch_eg_loss.append(enc_gen_train_on_batch(x, z, valid))

            cx_loss = np.mean(np.array(epoch_cx_loss), axis=0)
            cz_loss = np.mean(np.array(epoch_cz_loss), axis=0)
            eg_loss = np.mean(np.array(epoch_eg_loss), axis=0)            
            obj_loss = np.sum([cx_loss, cz_loss, eg_loss])
            
            tb_loss([cx_loss, cz_loss, eg_loss])
            train_loss.append([epoch, epochs, cx_loss, cz_loss, eg_loss])
            
            if epoch%check_point == 0:
                """수정사항: log 별도 저장 -> csv"""
                logging.info(
                    'Epoch: {}/{}, [Dx loss: {}] [Dz loss: {}] [G loss: {}]'.format(epoch, epochs, cx_loss, cz_loss, eg_loss)
                )

            with train_summary_writer.as_default():
                tf.summary.scalar('epoch_cx_loss', tb_epoch_cx_loss.result(), step=epoch)
                tf.summary.scalar('epoch_cz_loss', tb_epoch_cz_loss.result(), step=epoch)
                tf.summary.scalar('epoch_eg_loss', tb_epoch_eg_loss.result(), step=epoch)
                tf.summary.scalar('epoch_loss', tb_loss.result(), step=epoch)

            # critic_x.save(critic_x_model_path)
            # encoder.save(encoder_model_path)
            # generator.save(generator_model_path)
            # critic_z.save(critic_z_model_path)

            # critic_x_model.save(critic_x_m_model_path)
            # critic_z_model.save(critic_z_m_model_path)
            # encoder_generator_model.save(encoder_generator_model_path)
            train_dt=datetime.now().strftime("%Y-%m-%d")
            SaveModel(critic_x,'mongo_TadGAN','critic_x',train_dt)
            SaveModel(encoder,'mongo_TadGAN','encoder',train_dt)
            SaveModel(generator,'mongo_TadGAN','generator',train_dt)
            SaveModel(critic_z,'mongo_TadGAN','critic_z',train_dt)
            SaveModel(critic_x_model,'mongo_TadGAN','critic_x_model',train_dt)
            SaveModel(critic_z_model,'mongo_TadGAN','critic_z_model',train_dt)
            SaveModel(encoder_generator_model,'mongo_TadGAN','encoder_generator_model',train_dt)
            
            os.makedirs('./log/{}'.format(train_dt), exist_ok=True)            
            pd.DataFrame(train_loss, 
                         columns=['epoch','epochs','Dx_loss','Dz_loss','G_loss']
                        ).to_csv('./log/{}/train_loss_{}.csv'.format(train_dt, train_column))



    
    print("hello teng")

if __name__ == "__main__":
    print("entering main")
    print(sys.argv[1])
    if sys.argv[1] == 'iqr':
        print("entering iqr")
        iqr_mds_gan()
    elif sys.argv[1] == 'lstm':
        print("entering lstm")
        lstm_autoencoder()
    elif sys.argv[1] == 'oc_svm':
        print("entering svm")
        oc_svm()
    elif sys.argv[1] == 'tad_gan':
        print("entering tadgan")
        teng()
    elif sys.argv[1] == 'eval':
        print("entering data evaluation")
        data_eval()
    elif sys.argv[1] == 'infer_tad':
        print("entering inference tadgan")
        infer_tad()
    elif sys.argv[1] == 'infer_main':
        print("entering inference main product ")
        infer_main()
    elif sys.argv[1] == 'infer_vari':
        print("entering inference vari product ")
        infer_vari()
    print("hello main")
 
