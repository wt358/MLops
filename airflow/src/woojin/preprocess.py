import os
import re
import sys

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import MDS

from sklearn.cluster import DBSCAN

def customize(dataframe,mds_matrix):
    answer=pd.DataFrame(np.zeros(len(mds_matrix)))
    answer.rename(columns = {0:'Class'},inplace=True)## 일단 전부 양품으로 간주하느 데이터프레임 생성
    for i in range(1,10):
        for j in range(1,10):
            dbscan=DBSCAN(eps = i,min_samples=j)
            clusters_mds = pd.DataFrame(dbscan.fit_predict(mds_matrix))
            clusters_mds.rename(columns = {0:'Class'},inplace=True)
            if clusters_mds.value_counts().count()==2:
                if len(clusters_mds.loc[clusters_mds['Class'] == -1]) >  len(answer.loc[answer['Class'] == -1]): 
                    answer=clusters_mds
    dataframe.reset_index(inplace=True,drop=True)          
    result = pd.DataFrame(pd.concat([dataframe,answer], axis = 1))       
    return result

def IQR(section):
    
    section.iloc[:,6:17]=section.iloc[:,6:17].apply(pd.to_numeric,errors='coerce')
    for i in range(6,17):
        level_1q = section.iloc[:,i].quantile(0.025)
        level_3q = section.iloc[:,i].quantile(0.975)
        IQR = level_3q - level_1q
        rev_range = 1.5 # 제거 범위 조절 변수
        section = section[(section.iloc[:,i] <= level_3q + (rev_range * IQR)) & (section.iloc[:,i] >= level_1q - (rev_range * IQR))] ## sectiond에 저장된 데이터 프레임의 이상치 제거 작업
    return section

def MDS_molding(pds):
    list1=[]## 불량여부가 라벨링된 구간별 데이터 프레임을 저장할 리스트

    for i in range(len(pds)):
        start_time = time.time()
        print('%d 번째' %i)
        pds[i]['TimeStamp']=pd.to_datetime(pds[i]['TimeStamp']).astype('int64')/10**9
        dataframe=pds[i].iloc[:,5:]
        if len(pds[i])>=30:
            std = StandardScaler().fit_transform(pds[i].iloc[:,5:]) ## 정규화 진행
            end_time = time.time()
            print('    if std 코드 실행 시간: %20ds' % (end_time - start_time))
            mds_results = MDS(n_components=2).fit_transform(std) ## mds차원축소결과 저장(시간이 좀 많이 소요됨)
            end_time = time.time()
            print('    if mds 코드 실행 시간: %20ds' % (end_time - start_time))
            mds_results=pd.DataFrame(mds_results) ##dataframe 형태로 저장 
            end_time = time.time()
            print('    if df 코드 실행 시간: %20ds' % (end_time - start_time))
            list1.append(customize(pds[i],mds_results))## 구간별 라벨링 데이터 프레임을 리스트에 저장
            end_time = time.time()
            print('    if 코드 실행 시간: %20ds' % (end_time - start_time))
        else :
            answer=pd.DataFrame(np.zeros(len(pds[i]))) 
            answer.rename(columns = {0:'Class'},inplace=True) 
            dataframe.reset_index(inplace=True,drop=True)     
            result = pd.DataFrame(pd.concat([pds[i].iloc[:,5:],answer], axis = 1))
            list1.append(result)
            end_time = time.time()
            print('    else 코드 실행 시간: %20ds' % (end_time - start_time))
    df_all=pd.concat(list1, ignore_index=True)
    
    return df_all

def time_segments_aggregate(X, interval, time_column, method=['mean']):
    """Aggregate values over given time span.
    Args:
        X (ndarray or pandas.DataFrame):
            N-dimensional sequence of values.
        interval (int):
            Integer denoting time span to compute aggregation of.
        time_column (int):
            Column of X that contains time values.
        method (str or list):
            Optional. String describing aggregation method or list of strings describing multiple
            aggregation methods. If not given, `mean` is used.
    Returns:
        ndarray, ndarray:
            * Sequence of aggregated values, one column for each aggregation method.
            * Sequence of index values (first index of each aggregated segment).
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X = X.sort_values(time_column).set_index(time_column)

    if isinstance(method, str):
        method = [method]

    start_ts = X.index.values[0]
    max_ts = X.index.values[-1]

    values = list()
    index = list()
    while start_ts <= max_ts:
        end_ts = start_ts + interval
        subset = X.loc[start_ts:end_ts - 1]
        aggregated = [
            getattr(subset, agg)(skipna=True).values
            for agg in method
        ]
        values.append(np.concatenate(aggregated))
        index.append(start_ts)
        start_ts = end_ts

    return np.asarray(values), np.asarray(index)

def rolling_window_sequences(X, index, window_size, target_size, step_size, target_column,
                             drop=None, drop_windows=False):
    """Create rolling window sequences out of time series data.
    The function creates an array of input sequences and an array of target sequences by rolling
    over the input sequence with a specified window.
    Optionally, certain values can be dropped from the sequences.
    Args:
        X (ndarray):
            N-dimensional sequence to iterate over.
        index (ndarray):
            Array containing the index values of X.
        window_size (int):
            Length of the input sequences.
        target_size (int):
            Length of the target sequences.
        step_size (int):
            Indicating the number of steps to move the window forward each round.
        target_column (int):
            Indicating which column of X is the target.
        drop (ndarray or None or str or float or bool):
            Optional. Array of boolean values indicating which values of X are invalid, or value
            indicating which value should be dropped. If not given, `None` is used.
        drop_windows (bool):
            Optional. Indicates whether the dropping functionality should be enabled. If not
            given, `False` is used.
    Returns:
        ndarray, ndarray, ndarray, ndarray:
            * input sequences.
            * target sequences.
            * first index value of each input sequence.
            * first index value of each target sequence.
    """
    out_X = list()
    out_y = list()
    X_index = list()
    y_index = list()
    target = X[:, target_column]

    if drop_windows:
        if hasattr(drop, '__len__') and (not isinstance(drop, str)):
            if len(drop) != len(X):
                raise Exception('Arrays `drop` and `X` must be of the same length.')
        else:
            if isinstance(drop, float) and np.isnan(drop):
                drop = np.isnan(X)
            else:
                drop = X == drop

    start = 0
    max_start = len(X) - window_size - target_size + 1
    while start < max_start:
        end = start + window_size

        if drop_windows:
            drop_window = drop[start:end + target_size]
            to_drop = np.where(drop_window)[0]
            if to_drop.size:
                start += to_drop[-1] + 1
                continue

        out_X.append(X[start:end])
        out_y.append(target[end:end + target_size])
        X_index.append(index[start])
        y_index.append(index[end])
        start = start + step_size

    return np.asarray(out_X), np.asarray(out_y), np.asarray(X_index), np.asarray(y_index)
    
def preprocess(dataset:pd.DataFrame, time_columns:str) -> dict:
    trainset = {}
    for col in dataset.columns.tolist():
        d = dataset[col].reset_index().rename(columns={time_columns:'timestamp',col:'value'})
        d['timestamp'] = d['timestamp'].map(lambda x: x.timestamp()).astype(int)
        d['value'] = d['value'].astype(float)
        trainset[col] = d
    return trainset


def k_to_1000(t):
    if type(t) == str:
        if ('K' in t) or ('k' in t):
            t = re.sub('K|k','',t).strip()
            t = float(t)*1000
    else:
        pass
    return float(t)

def data_reshape(data_path, time_columns=None, vib_columns = ['x','y','z']):
    df = pd.read_csv(data_path)
    
    if time_columns == None:
        time_columns = df.dtypes[df.dtypes != float].index.tolist()[0]# params['time_columns']
        
    df[vib_columns] = df[vib_columns].apply(lambda col: col.map(k_to_1000))
    data = df.set_index(time_columns)
    if type(data.index[0]) == str:
        data.index = pd.to_datetime(data.index)
    dataset = preprocess(data, time_columns)
    return dataset



#     if ((df.shape[1] == 2)
#     if ((df.shape[1] == 2) & (df.columns[0] == 'timestamp') & (df.columns[1] == 'value')):
#         print('Do not reconstruct data from')
#         dataset = df.copy()

#     if not ((df.shape[1] == 2) & (df.columns[0] == 'timestamp') & (df.columns[1] == 'value')):
#     print('Reconstruct data form for TadGAN')

def outlier_iqr(df):
    scaled = []
    q = [0.05, 0.95]
    scaler= MinMaxScaler()
    print(df.columns)
    print(df.dtypes)
    for i in df.columns[1:]:
        lb, ub = df[i].quantile(q).tolist()
        samp = df[i].map(lambda x: None if (x < lb) or (x > ub) else x)
        samp = scaler.fit_transform(samp.to_frame())
        scaled.append(samp.reshape(-1,).tolist())
    df_scaled = pd.DataFrame(scaled).T
    df_scaled.columns = df.columns[1:]
    df_scaled.index = pd.to_datetime(df['TimeStamp'])
    return df_scaled, scaled
