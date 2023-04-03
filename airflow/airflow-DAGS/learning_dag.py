from datetime import datetime, timedelta

from kubernetes.client import models as k8s
from airflow.models import DAG
from airflow.models.variable import Variable

from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.kubernetes.secret import Secret
from airflow.kubernetes.pod import Resources
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
from pymongo import MongoClient, ASCENDING, DESCENDING
import pymongo
import pandas as pd

gpu_tag='0.15'
tad_tag='0.01'

dag_id = 'learning-dag'

molding_brand_name = ['WooJin', 'DongShin']
woojin_factory_name = ['NewSeoGwang', ]
dongshin_factory_name = ['teng',  ]

task_default_args = {
        'owner': 'coops2',
        'retries': 0,
        'retry_delay': timedelta(minutes=1),
        'depends_on_past': False,
        #'execution_timeout': timedelta(minutes=5)
}

dag = DAG(
        dag_id=dag_id,
        description='kubernetes pod operator',
        start_date=days_ago(8),
        default_args=task_default_args,
        schedule_interval=timedelta(days=7),
        max_active_runs=3,
        # catchup=True,
        catchup=False,
)
'''
env_from = [
        k8s.V1EnvFromSource(
            # configmap fields를  key-value 형태의 dict 타입으로 전달한다. 
            config_map_ref=k8s.V1ConfigMapEnvSource(name="airflow-cluster-pod-template"),
            # secret fields를  key-value 형태의 dict 타입으로 전달한다.
            secret_ref=k8s.V1SecretEnvSource(name="regcred")),
]

'''

configmaps = [
        k8s.V1EnvFromSource(config_map_ref=k8s.V1ConfigMapEnvSource(name='airflow-cluster-pod-template'))
        ]
'''
pod_resources = Resources()
pod_resources.limit_gpu = '1'
'''

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
# secret_all = Secret('env', None, 'db-secret-hk8b2hk77m')
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
secret_newseogwang = Secret('env', None, 'newseogwang-db-secret-d64ch8k9gf')
secret_teng= Secret('env', None, 'teng-db-secret-hk8b2hk77m')


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

paths=['path_main','path_vari']

def print_rank(df,i,machine_no,factory):
    today=pd.Timestamp.today()
    date_1month=(today- pd.DateOffset(months=i)).strftime('%Y-%m-%d %I:%M:%S')
    today=datetime.now().strftime("%Y-%m-%d")
    host = Variable.get("WOOJIN_MONGO_URL_SECRET")
    client = MongoClient(host)
    db_rank= client['rank']
    collection = db_rank[f'rank_{factory}_{machine_no}_{i}_{today}']
    collection.create_index([("rank",pymongo.ASCENDING)],unique=True)

    df2=df[df['TimeStamp'] > date_1month ]['Additional_Info_1'].value_counts()
    df1=df2.rank(method='min',ascending=False)
    print("\n",i,"month rank")
    print("====================================")
    df1=df1.rename("rank")
    df2=df2.rename("count")
    df_new=pd.concat([df1,df2],axis=1).reset_index()
    print(df_new)
    data=df_new.to_dict('records')
    try:
        collection.insert_many(data,ordered=False)
    except Exception as e:
        print("mongo connection failer",e)
    print("====================================")
    client.close()
    return

def print_stat_woojin(df,machine_no,factory):
    host = Variable.get("WOOJIN_MONGO_URL_SECRET")
    client = MongoClient(host)
    db_rank= client['stat']
    today=pd.Timestamp.today()
    date_1month=(today- pd.DateOffset(months=1)).strftime('%Y-%m-%d %I:%M:%S')
    mode_machine_name=df[df['TimeStamp'] > date_1month ]['Additional_Info_1'].value_counts().idxmax()
    print("main product ",mode_machine_name)
    df= df[df['Additional_Info_1'] == mode_machine_name]
    df.drop(columns={'idx','Machine_Name','Additional_Info_1', 'Additional_Info_2','Shot_Number',
        },inplace=True)
    print(df)
    
    for i in [1,6]:
        today=pd.Timestamp.today()
        date_1month=(today- pd.DateOffset(months=i)).strftime('%Y-%m-%d %I:%M:%S')
        print(date_1month)
        today=datetime.now().strftime("%Y-%m-%d")
        collection = db_rank[f'stat_{factory}_{machine_no}_{i}month_{today}']
        collection.create_index([("Today",pymongo.ASCENDING),("Feature",pymongo.ASCENDING)],unique=True)
        df2=df[df['TimeStamp'] > date_1month ]
        print(df2)
        stat_df=df2.drop(columns={'TimeStamp'}).describe().T
        
        # df1=df2.rank(method='min',ascending=False)
        
        print("\n",i,"month stat")
        print("====================================")
        stat_df.reset_index(inplace=True)
        stat_df = stat_df.rename(columns = {'index':'Feature'})
        print(stat_df)
        # df1=df1.rename("rank")
        # df2=df2.rename("count")
        # df_new=pd.concat([df1,df2],axis=1).reset_index()
        # print(df_new)
        stat_df['Today']=today
        data=stat_df.to_dict('records')
        try:
            collection.insert_many(data,ordered=False)
        except Exception as e:
            print("mongo connection failer",e)
        print("====================================")
    client.close()
    return


def print_stat(df,i,factory):
    today=pd.Timestamp.today()
    date_1month=(today- pd.DateOffset(months=i)).strftime('%Y-%m-%d %I:%M:%S')
    today=datetime.now().strftime("%Y-%m-%d")
    host = Variable.get("MONGO_URL_SECRET")
    client = MongoClient(host)
    db_rank= client['stat']
    collection = db_rank[f'{factory}_{i}month_{today}']
    collection.create_index([("Today",ASCENDING),("Feature",ASCENDING)],unique=True)
    
    print(df)
    
    df2=df[df['_time'] > date_1month ]
    print(df2)
    if df2.empty:
        print(f"there is no data in {i} month")
        return 1
    print("\n",i,"month statistics")
    print("====================================")
    stat_df=df2.drop(columns={'_time'}).describe().T
    stat_df.reset_index(inplace=True)
    stat_df=stat_df.rename(columns={'index':'Feature'})
    print(stat_df)
    
    stat_df['Today']=today
    data=stat_df.to_dict('records')
    try:
        collection.insert_many(data,ordered=False)
    except Exception as e:
        print("mongo connection failer",e)
    print("====================================")
    client.close()
    return


def which_path_dongshin(**kwargs):
    '''
    return the task_id which to be executed
    '''
    factory=kwargs['factory_name']
    print(factory)
    host=Variable.get("MONGO_URL_SECRET")
    client=MongoClient(host)
    db=client["raw_data"]
    collection=db[f"{factory}_mold_data"]

    now=datetime.now()
    start=now-timedelta(days=183)
    query={
            '_time':{
                '$gt':start,
                '$lt':now
                }
            }
    try:
        df = pd.DataFrame(list(collection.find(query)))
    except Exception as e:
        print("mongo connection failer during pull",e)
    df.drop(columns={'_id','_time','result','_measurement','table','_start','_stop'},inplace=True)
    df.rename(columns={"idx":"_time"},inplace=True)
    month_list = [1,6]
    print("======================================================")
    print("  6호기")
    for i in month_list:
        print_stat(df, i,factory)
    print("======================================================")
    client.close()
#   if '9000a' in mode_machine_name:
    if True:
        task_id = 'path_main'
    else:
        task_id = 'path_vari'
    return task_id+'_'+factory

def which_path_woojin(**kwargs):
    '''
    return the task_id which to be executed
    '''
    factory=kwargs['factory_name']
    print(factory)
    print(factory)
    host = Variable.get("WOOJIN_MS_HOST")
    database = Variable.get("WOOJIN_MS_DATABASE")
    username = Variable.get("WOOJIN_MS_USERNAME")
    password = Variable.get("WOOJIN_MS_PASSWORD")

    query = text(
        "SELECT * from shot_data WITH(NOLOCK) where TimeStamp > DATEADD(day,-7,GETDATE())"
        )
    query1 = text(
        "SELECT * from shot_data WITH(NOLOCK) where TimeStamp > DATEADD(month,-6,GETDATE())"
        )
    conection_url = sqlalchemy.engine.url.URL(
        drivername="mssql+pymssql",
        username=username,
        password=password,
        host=host,
        database=database,
    )
    engine = create_engine(conection_url, echo=True)
    
    sql_result_pd = pd.read_sql_query(query, engine)
    sql_result_pd['Additional_Info_1']=sql_result_pd['Additional_Info_1'].str.strip()
    mode_machine_name=sql_result_pd['Additional_Info_1'].value_counts().idxmax()
    print(sql_result_pd['Additional_Info_1'].value_counts())
    print(sql_result_pd)
    print(mode_machine_name)
    
    sql_result = engine.execute(query1)
    sql_result_pd = pd.read_sql_query(query1, engine)
    sql_result_pd['Additional_Info_1']=sql_result_pd['Additional_Info_1'].str.strip()
    sql_result_pd = sql_result_pd[sql_result_pd['Machine_Name'] != '7']
    sql_result_pd = sql_result_pd[sql_result_pd['Machine_Name'] != '6i']
    sql_result_pd_6 = sql_result_pd[sql_result_pd['Machine_Name'] != '']
    sql_result_pd_25 = sql_result_pd[sql_result_pd['Machine_Name'] == '']
    #   month_list = [1, 3, 6]
    month_list = [1,6]
    print("======================================================")
    print("  6호기")
    for i in month_list:
        print_rank(sql_result_pd_6, i,6,factory)
    print_stat_woojin(sql_result_pd_6,6,factory)
    print("======================================================")
    print("  25호기")
    for i in month_list:
        print_rank(sql_result_pd_25, i,25,factory)
    print_stat_woojin(sql_result_pd_25,25,factory)
    print("======================================================")
    engine.dispose()
    if '9000a' in mode_machine_name:
    # if True:
        task_id = 'path_main'
    else:
        task_id = 'path_vari'
    return task_id+'_'+factory


for i in molding_brand_name:
    i=i.lower()
    fact=f'{i}_factory_name'
    fact_list=eval(fact)
    for j in fact_list:
        j=j.lower()
        start = DummyOperator(task_id=f"start_{j}", dag=dag)
        run_iqr = KubernetesPodOperator(
                task_id="iqr_gan_pod_operator_"+j,
                name="iqr-gan",
                namespace='airflow-cluster',
                image=f'ctf-mlops.kr.ncr.ntruss.com/cuda:{gpu_tag}',
                #image_pull_policy="Always",
                image_pull_secrets=[k8s.V1LocalObjectReference('regcred')],
                cmds=["sh" ],
                arguments=["command.sh",i, "iqr"],
                affinity=gpu_aff,
                #resources=pod_resources,
                secrets=[eval('secret_'+j),secret_all1 ,secret_all2 ,secret_all3, secret_all4, secret_all5, secret_all6, secret_all7, secret_all8,  secret_alla, secret_allb ],
                env_vars={'EXECUTION_DATE':"{{ds}}"},
                #env_vars={'MONGO_URL_SECRET':'{{var.value.MONGO_URL_SECRET}}'},
                #configmaps=configmaps,
                is_delete_operator_pod=True,
                get_logs=True,
                startup_timeout_seconds=600,
                )

        run_lstm = KubernetesPodOperator(
                task_id="lstm_pod_operator_"+j,
                name="lstm-auto-encoder",
                namespace='airflow-cluster',
                image=f'ctf-mlops.kr.ncr.ntruss.com/cuda:{gpu_tag}',
                #image_pull_policy="Always",
                #image_pull_policy="IfNotPresent",
                image_pull_secrets=[k8s.V1LocalObjectReference('regcred')],
                cmds=["sh" ],
                arguments=["command.sh",i, "lstm"],
                affinity=gpu_aff,
                #resources=pod_resources,
                secrets=[eval('secret_'+j),secret_all1 ,secret_all2 ,secret_all3, secret_all4, secret_all5, secret_all6, secret_all7, secret_all8,  secret_alla, secret_allb ],
                env_vars={'EXECUTION_DATE':"{{ds}}"},
                #env_vars={'MONGO_URL_SECRET':'/var/secrets/db/mongo-url-secret.json'},
                #configmaps=configmaps,
                is_delete_operator_pod=True,
                get_logs=True,
                startup_timeout_seconds=600,
                )
        run_tadgan = KubernetesPodOperator(
                task_id="tad_pod_operator_"+j,
                name="tad-gan",
                namespace='airflow-cluster',
                image=f'ctf-mlops.kr.ncr.ntruss.com/tad:{tad_tag}',
                #image_pull_policy="Always",
                #image_pull_policy="IfNotPresent",
                image_pull_secrets=[k8s.V1LocalObjectReference('regcred')],
                cmds=["sh" ],
                arguments=["command.sh",i, "tad_gan"],
                affinity=gpu_aff,
                #resources=pod_resources,
                secrets=[eval('secret_'+j),secret_all1 ,secret_all2 ,secret_all3, secret_all4, secret_all5, secret_all6, secret_all7, secret_all8,  secret_alla, secret_allb ],
                env_vars={'EXECUTION_DATE':"{{ds}}"},
                #env_vars={'MONGO_URL_SECRET':'/var/secrets/db/mongo-url-secret.json'},
                #configmaps=configmaps,
                is_delete_operator_pod=True,
                get_logs=True,
                startup_timeout_seconds=600,
                )
        run_svm = KubernetesPodOperator(
                task_id="oc_svm_pod_operator_"+j,
                name="oc-svm",
                namespace='airflow-cluster',
                image=f'ctf-mlops.kr.ncr.ntruss.com/cuda:{gpu_tag}',
                #image_pull_policy="Always",
                image_pull_secrets=[k8s.V1LocalObjectReference('regcred')],
                secrets=[eval('secret_'+j),secret_all1 ,secret_all2 ,secret_all3, secret_all4, secret_all5, secret_all6, secret_all7, secret_all8,  secret_alla, secret_allb ],
                env_vars={'EXECUTION_DATE':"{{ds}}"},
                cmds=["sh" ],
                arguments=["command.sh",i,"oc_svm"],
                affinity=cpu_aff,
                is_delete_operator_pod=True,
                get_logs=True,
                startup_timeout_seconds=600,
                )
        run_eval = KubernetesPodOperator(
                task_id="eval_pod_operator_"+j,
                name="data-eval",
                namespace='airflow-cluster',
                image=f'ctf-mlops.kr.ncr.ntruss.com/cuda:{gpu_tag}',
                #image_pull_policy="Always",
                image_pull_secrets=[k8s.V1LocalObjectReference('regcred')],
                secrets=[eval('secret_'+j),secret_all1 ,secret_all2 ,secret_all3, secret_all4, secret_all5, secret_all6, secret_all7, secret_all8,  secret_alla, secret_allb ],
                env_vars={'EXECUTION_DATE':"{{ds}}"},
                cmds=["sh" ],
                arguments=["command.sh",i,"eval"],
                affinity=cpu_aff,
                is_delete_operator_pod=True,
                get_logs=True,
                startup_timeout_seconds=600,
                )



        main_or_vari = BranchPythonOperator(
            task_id = 'branch_'+j,
            python_callable=eval('which_path_'+i),
            op_kwargs={'brand_name':i,'factory_name':j},
            dag=dag,
        )

        after_aug = DummyOperator(task_id="Aug_fin_"+j, dag=dag)
        after_ml = DummyOperator(task_id="ML_fin_"+j, dag=dag)

        start >> main_or_vari

        for path in paths:
            t = DummyOperator(
                task_id=path+'_'+j,
                dag=dag,
                )
            
            if path == 'path_main':
                main_or_vari >> t >> run_iqr >> after_aug 
                after_aug >> [run_svm, run_lstm] >> after_ml
                after_aug >> run_eval
            elif path == 'path_vari':
                # main_or_vari >> t >> run_tadgan
                main_or_vari >> t 
                
