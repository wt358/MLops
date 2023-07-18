
from datetime import timedelta
from datetime import datetime

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models.variable import Variable
from airflow.utils.trigger_rule import TriggerRule
from scipy.stats import kstest, normaltest, shapiro, anderson

import numpy as np
import pandas as pd
import time
from pymongo import MongoClient
import matplotlib.pyplot as plt
import matplotlib

args = {
    'owner': 'airflow',
}

host = Variable.get("UCT_MONGO_URL_SECRET")
client=MongoClient(host)
db=client['raw_db']
collection=db['limit_ttl']

try:
    limt_ttl=list(collection.find())[0]
except Exception as e:
    print(e)
print(limt_ttl)
print(type(limt_ttl))

def boxplot_whiskers(x):
    plt.figure()
    boxplot = plt.boxplot(x)
    boxplot_whiskers = [item.get_ydata() for item in boxplot['whiskers']]
    return(boxplot_whiskers)

def outlier_check(x):
    y = boxplot_whiskers(x)
    print(np.min(y), np.max(y))
    sigma_3_p = np.mean(x) + 3*np.std(x) 
    sigma_3_m = np.mean(x) - 3*np.std(x)
    print('3 sigma range :',sigma_3_m, sigma_3_p)
    out_idx = np.where(x>=sigma_3_p)[0]
    out_values = x[out_idx]
    # plt.figure()
    # boxplot = plt.boxplot(x[-out_idx])
    return(pd.DataFrame({'outlier_index': out_idx,'outliers': out_values}))


def Significance_Test(df_in):
    
    test_stat, p_val = shapiro(df_in)
    ks_stat, p_valks = kstest(df_in, 'norm')
    nm_stat, p_valnm = normaltest(df_in)

    print("SHAPIRO Result : test-statistics : {}, p-value : {}".format(test_stat, p_val))
    print("KS Result : test-statistics : {}, p-value : {}".format(ks_stat, p_valks))
    print("Normal Result : test-statistics : {}, p-value : {}".format(nm_stat, p_valnm))

    test_res1 = anderson(df_in, dist = 'norm')
    print(test_res1)
    
    
def statistic_anomaly_detection(df_in, process, upper_sig=2, under_sig=3):
    mean = df_in[process].mean()
    std = df_in[process].std()
    test_des = df_in[(df_in[process] < (mean+(3*std))) & (df_in[process] > (mean-(3*std)))][process].describe()
    print(test_des)
    print()
    print()
    print('mean ', test_des['mean'])

    usl, lsl = test_des['mean'] + (upper_sig*test_des['std']), test_des['mean'] - (under_sig*test_des['std'])
    ucl, lcl = limt_ttl[process]['ucl'], limt_ttl[process]['lcl']
    print(f"usl: {usl}, lsl: {lsl}")
    print(f"ucl: {ucl}, lcl: {lcl}")

    print()
    print()
    test_df_c = df_in.copy()


    anomal_upper = test_df_c[test_df_c[process]>= usl].index
    anomal_under = test_df_c[test_df_c[process]<= lsl].index
    test_df_c['anomal_detection'] = 0
    test_df_c.loc[anomal_upper, 'anomal_detection'] = 2
    test_df_c.loc[anomal_under, 'anomal_detection'] =1
    test_df_c.reset_index(drop=True, inplace=True)
    print(test_df_c)



    print()
    print()
    mask = test_df_c['anomal_detection'].values
    print(mask)


def analy_all():
    
    sig_values = [('DC12V 전원 검사',2,3),('DC 5V 전원 검사',0.9,1),('SENSOR0 (TRA)',6,6),('SENSOR1 (TDF)',6,6),
                ('DISP MOSI HIGH',0.2,2),('DISP MISO HIGH',2,6),('DISP CLK HIGH',2,6),('DISP STB HIGH',2,6)]
    for process,upper,under in sig_values:
        check_valid(process=process,upper=upper,under=under)


def check_valid(process,upper,under):
    
    host = Variable.get("UCT_MONGO_URL_SECRET")
    client=MongoClient(host)
    db=client['raw_db']
    collection=db['com_ttl']
   
    try:
        df=pd.DataFrame(list(collection.find()))
    except Exception as e:
        print(e)
    df.Status.value_counts()
    df.Status.isna().sum()
    df.dropna(subset=['Status'],axis=0, inplace=True)
    df['COM0 MODE TTL'].unique()
    df.loc[df[(df['Status']=='CN3')|(df['Status']=='K1.1')|(df['Status']=='JP1.1')|(df['Status']=='0 : NO TEST 1 : TEST 진행중 2 : TEST 완료 ')].index, 'COM0 MODE TTL']=0
    df['COM0 MODE TTL'].fillna(1, inplace=True)
    df['COM0 MODE TTL'].unique()
    df.drop(['date.1','qty.1'], axis=1, inplace=True)
    df.Status.unique()

    df_list = dict()
    numeric = ['DC12V 전원 검사', 'DC 5V 전원 검사', 'SENSOR0 (TRA)', 
           'SENSOR1 (TDF)', 'DISP MOSI HIGH', 'DISP MISO HIGH', 'DISP CLK HIGH', 'DISP STB HIGH']

    defect = ['K1.1','JP1.1','CN5.1(19.8KΩ), 3.046Volt', 'CN6.1(25.2KΩ), 3.328Volt', 'CN9.3, DC전압 검출', 'CN9.4, DC전압 검출', 'CN9.5, DC전압 검출', 'CN9.6, DC전압 검출']

    for n, d in zip(numeric, defect):
        df_list[n] = pd.DataFrame(df[(df['Status']==d) | (df['Status']=='OK')][['date','qty',n, 'Status']]).reset_index(drop=True)


    df_new = df_list[process]
    print(df_new.isna().sum())
    print(df_new[df_new[process].isna()==True])
    ## na가 확인 되므로 처리해야한다.
    ## 특히 해당 검사에서 문제가 발생된 항목의 NA이므로 OK가 아닌 값들의 평균으로 채워 준다.
    na_v = df_new[df_new['Status']!='OK'][process].mean() # OK가 아닌 값들의 평균 구하기
    df_new[process].fillna(na_v,inplace = True) # 구한 평균으로 NA자리를 채운다.

    Significance_Test(df_in=df_new[process])

    outlier_check(df_new[process].values)
    print(limt_ttl[process])
    statistic_anomaly_detection(df_in=df_new, process=process, upper_sig=upper, under_sig=under)


def anal_gan():
    time.sleep(234)

with DAG(
    dag_id='uct_pipeline',
    default_args=args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=['example'],
) as dag:

    dummy1 = DummyOperator(task_id="start")
    dummy2 = DummyOperator(task_id="augmentation_finised")
    dummy3 = DummyOperator(task_id="analysis_finished")
    
    analy = PythonOperator(
        task_id="anal_all",
        python_callable=analy_all,
        # depends_on_past=True,
        depends_on_past=False,
        owner="coops2",
        retries=0,
        retry_delay=timedelta(minutes=1),
    )
    gan = PythonOperator(
        task_id="augmentation",
        python_callable=anal_gan,
        # depends_on_past=True,
        depends_on_past=False,
        owner="coops2",
        retries=0,
        retry_delay=timedelta(minutes=1),
    )
    dummy1>>gan>>dummy2>>analy>>dummy3