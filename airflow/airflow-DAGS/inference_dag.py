from datetime import timedelta
from datetime import datetime
from kubernetes.client import models as k8s
from airflow.kubernetes.secret import Secret
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models.variable import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score, classification_report, plot_confusion_matrix, confusion_matrix

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM 
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns


from bson import ObjectId

import gridfs
import io

from gridfs import GridFS



from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate, Dropout, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.python.client import device_lib
from sklearn.utils import shuffle
import tensorflow as tf

import joblib


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import influxdb_client
import csv
import pandas as pd
import os
import time
import numpy as np
from collections import Counter
from kafka import KafkaConsumer
from kafka import KafkaProducer
from pymongo import MongoClient, ASCENDING, DESCENDING

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

from json import loads
import random as rn

gpu_tag='0.14'
tad_tag='0.01'

np.random.seed(34)

# manual parameters
RANDOM_SEED = 42
TRAINING_SAMPLE = 50000
VALIDATE_SIZE = 0.2

# setting random seeds for libraries to ensure reproducibility
np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

rn.seed(10)

secret_env = Secret(
        # Expose the secret as environment variable.
        deploy_type='env',
        # The name of the environment variable, since deploy_type is `env` rather
        # than `volume`.
        deploy_target='MONGO_URL_SECRET',
        # Name of the Kubernetes Secret
        secret='db-secret-hk8b2hk77m',
        # Key of a secret stored in this Secret object
        key='MONGO_URL_SECRET')
secret_volume = Secret(
        deploy_type='volume',
        # Path where we mount the secret as volume
        deploy_target='/var/secrets/db',
        # Name of Kubernetes Secret
        secret='db-secret-hk8b2hk77m',
        # Key in the form of service account file name
        key='mongo-url-secret.json')
secret_all = Secret('env', None, 'db-secret-hk8b2hk77m')
secret_all1 = Secret('env', None, 'airflow-cluster-config-envs')
secret_all2 = Secret('env', None, 'airflow-cluster-db-migrations')
secret_all3 = Secret('env', None, 'airflow-cluster-pgbouncer')
secret_all4 = Secret('env', None, 'airflow-cluster-pgbouncer-certs')
secret_all5 = Secret('env', None, 'airflow-cluster-postgresql')
secret_all6 = Secret('env', None, 'airflow-cluster-sync-users')
secret_all7 = Secret('env', None, 'airflow-cluster-token-7wptr')
secret_all8 = Secret('env', None, 'airflow-cluster-webserver-config')
secret_alla = Secret('env', None, 'airflow-ssh-git-secret')
secret_allb = Secret('env', None, 'default-token-8d2dz')


gpu_aff={
        'nodeAffinity': {
            # requiredDuringSchedulingIgnoredDuringExecution means in order
            # for a pod to be scheduled on a node, the node must have the
            # specified labels. However, if labels on a node change at
            # runtime such that the affinity rules on a pod are no longer
            # met, the pod will still continue to run on the node.
            'requiredDuringSchedulingIgnoredDuringExecution': {
                'nodeSelectorTerms': [{
                    'matchExpressions': [{
                        # When nodepools are created in Google Kubernetes
                        # Engine, the nodes inside of that nodepool are
                        # automatically assigned the label
                        # 'cloud.google.com/gke-nodepool' with the value of
                        # the nodepool's name.
                        'key': 'kubernetes.io/hostname',
                        'operator': 'In',
                        # The label key's value that pods can be scheduled
                        # on.
                        'values': [
                            'pseudo-gpu-w-2bsh',
                            #'pool-1',
                            ]
                        }]
                    }]
                }
            }
        }
cpu_aff={
        'nodeAffinity': {
            'requiredDuringSchedulingIgnoredDuringExecution': {
                'nodeSelectorTerms': [{
                    'matchExpressions': [{
                        'key': 'kubernetes.io/hostname',
                        'operator': 'In',
                        'values': [
                            'high-memory-w-23op',
                            'high-memory-w-23oq',
                            ]
                        }]
                    }]
                }
            }
        }



tf.random.set_seed(10)
class ModelSingleton(type):
   """
   Metaclass that creates a Singleton base type when called.
   """
   _mongo_id = {}
   def __call__(cls, *args, **kwargs):
       mongo_id = kwargs['mongo_id']
       if mongo_id not in cls._mongo_id:
           print('Adding model into ModelSingleton')
           cls._mongo_id[mongo_id] = super(ModelSingleton, cls).__call__(*args, **kwargs)
       return cls._mongo_id[mongo_id]

class LoadModel(metaclass=ModelSingleton):
   def __init__(self, *args, **kwargs):
       print(kwargs)
       self.mongo_id = kwargs['mongo_id']
       self.clf = self.load_model()
   def load_model(self):
       print('loading model')

       mongoClient = MongoClient()
       host = Variable.get("MONGO_URL_SECRET")
       client = MongoClient(host)

       db_model = client['coops2022_model']
       fs = gridfs.GridFS(db_model)
       print(self.mongo_id)
       f = fs.find({"_id": ObjectId(self.mongo_id)}).next()
       print(f)

       with open(f'{f.model_name}.joblib', 'wb') as outfile:
           outfile.write(f.read())
       return joblib.load(f'{f.model_name}.joblib')

# define funcs
def model_inference():
    #data consumer
    now = datetime.now()
    curr_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    consumer = KafkaConsumer('test.coops2022_etl.etl_data',
            group_id=f'Inference_model_{curr_time}',
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
    
    
    mongoClient = MongoClient()
    host = Variable.get("MONGO_URL_SECRET")
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
    
    X_test = scaler.transform(X_test)# ????????? scaler??? pull raw 2 aug?????? ????????? ???????????? ?????? ???????????? ??? ????????? ???????????? transform(X_test)??? ?????????.
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
    
    # ???????????? -1?????? ????????????.
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


    print("hello inference")

def push_onpremise():
    model_names = ['LSTM_autoencoder','OC_SVM']
    
    now=datetime.now()
    start=now-timedelta(days=50)
    for model_name in model_names:
            
        host = Variable.get("MONGO_URL_SECRET")
        client = MongoClient(host)
        db_result = client['result_log']
        collection = db_result[f'log_{model_name}_teng']
        query={
            'TimeStamp':{
                '$gt':start,
                '$lt':now
                }
            }
        try:
            df = pd.DataFrame(list(collection.find(query)))
        except Exception as e:
            print("mongo connection failer during pull",e)
        print(df)
        client.close()
        if df.shape[0]==0:
            print("empty")
            break
        df=df.drop_duplicates(subset=["_id"])
        df.drop(columns={'_id'},inplace=True)

        print(df.head())

    # for on premise
        host = Variable.get("LOCAL_MONGO_URL_SECRET")
        client = MongoClient(host)
        db_model = client['result_log']
        collection=db_model[f'teng_{model_name}']
        collection.create_index([("TimeStamp",ASCENDING)],unique=True)
        data=df.to_dict('records')

        try:
            collection.insert_many(data,ordered=False)
        except Exception as e:
            print("mongo connection failer during push",e)
        client.close()
    print("hello push on premise")

def which_path():
  '''
  return the task_id which to be executed
  '''
#   host = Variable.get("MS_HOST")
#   database = Variable.get("MS_DATABASE")
#   username = Variable.get("MS_USERNAME")
#   password = Variable.get("MS_PASSWORD")

#   query = text(
#       "SELECT * from shot_data WITH(NOLOCK) where TimeStamp > DATEADD(day,-7,GETDATE())"
#       )
#   conection_url = sqlalchemy.engine.url.URL(
#       drivername="mssql+pymssql",
#       username=username,
#       password=password,
#       host=host,
#       database=database,
#   )
#   engine = create_engine(conection_url, echo=True)
  
#   sql_result_pd = pd.read_sql_query(query, engine)
#   mode_machine_name=sql_result_pd['Additional_Info_1'].value_counts().idxmax()
#   print(sql_result_pd['Additional_Info_1'].value_counts())
#   print(mode_machine_name) 
  
  
#   if '9000a' in mode_machine_name:
  if True:
    task_id = 'path_main'
  else:
    task_id = 'path_vari'
  return task_id


paths=['path_main','path_vari']
# define DAG with 'with' phase
with DAG(
    dag_id="inference_dag", # DAG??? ???????????? ??????????????????.
    description="Model deploy and inference", # DAG??? ?????? ???????????????.
    start_date=days_ago(2), # DAG ?????? ?????? 2??? ????????? ???????????????.
    schedule_interval=timedelta(days=1), # ?????? 00:00??? ???????????????.
    tags=["inference"],
    max_active_runs=3,
    ) as dag:
    # t1 = PythonOperator(
    #     task_id="model_inference",
    #     python_callable=model_inference,
    #     depends_on_past=True,
    #     owner="coops2",
    #     retries=0,
    #     retry_delay=timedelta(minutes=1),
    # )
    t2 = PythonOperator(
        task_id="push_on_premise",
        python_callable=push_onpremise,
        depends_on_past=True,
        owner="coops2",
        retries=0,
        retry_delay=timedelta(minutes=1),
        trigger_rule='none_failed_min_one_success',
    )
    main_or_vari = BranchPythonOperator(
        task_id = 'branch',
        python_callable=which_path,
        dag=dag,
    )
    
    # infer_tadgan = KubernetesPodOperator(
    #     task_id="tad_infer_pod_operator",
    #     name="tad-infer-gan",
    #     namespace='airflow-cluster',
    #     image=f'ctf-mlops.kr.ncr.ntruss.com/tad:{tad_tag}',
    #     # image_pull_policy="Always",
    #     # image_pull_policy="IfNotPresent",
    #     image_pull_secrets=[k8s.V1LocalObjectReference('regcred')],
    #     cmds=["sh"],
    #     arguments=["command.sh", "infer_tad"],
    #     affinity=gpu_aff,
    #     # resources=pod_resources,
    #     secrets=[secret_all, secret_all1, secret_all2, secret_all3, secret_all4, secret_all5,
    #              secret_all6, secret_all7, secret_all8, secret_alla, secret_allb],
    #     env_vars={'EXECUTION_DATE':"{{ds}}"},
    #     # env_vars={'MONGO_URL_SECRET':'/var/secrets/db/mongo-url-secret.json'},
    #     # configmaps=configmaps,
    #     is_delete_operator_pod=True,
    #     get_logs=True,
    #     startup_timeout_seconds=600,
    # )
    infer_main = KubernetesPodOperator(
        task_id="main_infer_lstm_pod_operator",
        name="main-infer-lstm",
        namespace='airflow-cluster',
        image=f'ctf-mlops.kr.ncr.ntruss.com/cuda:{gpu_tag}',
        # image_pull_policy="Always",
        # image_pull_policy="IfNotPresent",
        image_pull_secrets=[k8s.V1LocalObjectReference('regcred')],
        cmds=["sh"],
        arguments=["command.sh", "infer_main"],
        affinity=gpu_aff,
        # resources=pod_resources,
        secrets=[secret_all, secret_all1, secret_all2, secret_all3, secret_all4, secret_all5,
                 secret_all6, secret_all7, secret_all8,  secret_alla, secret_allb],
        env_vars={'EXECUTION_DATE':"{{ds}}"},
        # env_vars={'MONGO_URL_SECRET':'/var/secrets/db/mongo-url-secret.json'},
        # configmaps=configmaps,
        is_delete_operator_pod=True,
        get_logs=True,
        startup_timeout_seconds=600,
    )
    
    infer_svm = KubernetesPodOperator(
        task_id="main_infer_svm_pod_operator",
        name="main-infer-ocsvm",
        namespace='airflow-cluster',
        image=f'ctf-mlops.kr.ncr.ntruss.com/cuda:{gpu_tag}',
        # image_pull_policy="Always",
        # image_pull_policy="IfNotPresent",
        image_pull_secrets=[k8s.V1LocalObjectReference('regcred')],
        cmds=["sh"],
        arguments=["command.sh", "infer_ocsvm"],
        affinity=cpu_aff,
        # resources=pod_resources,
        secrets=[secret_all, secret_all1, secret_all2, secret_all3, secret_all4, secret_all5,
                 secret_all6, secret_all7, secret_all8,  secret_alla, secret_allb],
        env_vars={'EXECUTION_DATE':"{{ds}}"},
        # env_vars={'MONGO_URL_SECRET':'/var/secrets/db/mongo-url-secret.json'},
        # configmaps=configmaps,
        is_delete_operator_pod=True,
        get_logs=True,
        startup_timeout_seconds=600,
    )

    dummy1 = DummyOperator(task_id="path1")
    # ????????? ????????? ????????????.
    # t1 ?????? ??? t2??? ???????????????.
    dummy1 >> main_or_vari
    
    for path in paths:
        t = DummyOperator(
            task_id=path,
            dag=dag,
            )
        
        if path == 'path_main':
            # main_or_vari>>t>>infer_main 
            main_or_vari>>t>>infer_main >> t2

        elif path == 'path_vari':
            # main_or_vari>>t>>infer_tadgan
            main_or_vari>>t>>infer_svm >> t2


