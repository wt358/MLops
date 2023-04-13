# MLops

## 1. 프로젝트 개요

이 프로젝트는 사출 공정 과정에서 사출기와 그 주변기기에서 생기는 데이터를 활용하여 MLops 구성하는 것을 목표로 한다.

데이터 품질 관리, ML model 관리 등의 기능을 완전 자동화함으로써 MLops를 구현했다. 각각의 모듈들은 airflow로 스케줄되어 관리된다. 

이 repository에는 airflow로 제어되는 DAG와 DAG를 실행하기 위해 필요한 소스코드들이 있다.

## 2. Developing Environment 

Develop Server : Ubuntu 20.04
Kubernetes : 1.23.9
airflow version : v2.2.5

### 2.1. Nodes' Specs

high-memory :  2 nodes(Auto-Scailing 1~4 nodes), [High Memory] vCPU 2EA, Memory 16GB , [SSD]Disk 50GB
pseudo-gpu : 1 node ,[Standard] vCPU 2EA, Memory 8GB , [SSD]Disk 50GB

## 3. DAGS

- ### 3.1. pull_raw_dag

사출기 이외 주변기기가 없는 경우, raw_data로 당겨온다.

사출기 이외 주변기기가 있는 경우, 사출기 데이터는 raw_data로 당겨오고, 주변기기 데이터는 peripheral_data로 당겨온다.

당겨온 데이터는 다시 pull_transform을 통해서 합쳐지고, 전처리 되어 etl_data에 들어간다.

- ### 3.2. learning_dag

이 DAG에서는 데이터 품질 개선, 모델 학습과 관련된 전반적인 task들이 실행된다.

Task 별로 GPU가 필요한 작업들이 있다. gan, LSTM auto-encoder 들이 그 예시이며, 해당 task들은 gpu가 존재하는 node에서 실행되어야 한다.

Task 별 다른 이미지와 node selection이 필요하기에 Kubernetes Pod Operator를 사용했다.

이 DAG에서 사용되는 소스들은 airflow/src/ 안에 사출기 브랜드별 폴더 안에 있고, entrypoint는 copy_gpu_py.py이다.

- #### 3.2.1 데이터 증강

사출기 데이터에서 anomaly를 detection 하기 위해 데이터를 모델에 학습을 시켜야 한다. 그런데 사출기 데이터 특성 상 대부분의 데이터는 양품이고 불량품 데이터는 극소수에 불과하다. Anomaly Detection에서는 이 불량품의 데이터에 대해서 학습하고 이를 구별해내야 하므로, 극소량의 불량품 데이터의 양을 증강시킬 필요가 있다.

이를 위해서 outlier 제거, labeling, clustering, Augmentation 과정이 아래 함수에서 수행된다.

    iqr_mds_gan()



- #### 3.2.2 데이터 품질 평가

데이터 증강이 끝나면 증강된 데이터에 대한 데이터 품질 평가가 시작된다. 

airflow/src/{사출기브랜드이름}/data_evaluation.py 안에 데이터 품질과 관련된 함수들이 저장되어 있다.

    classifiers = [
        KNeighborsClassifier(),
        SVC(),
        # GaussianProcessClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier(alpha=1, max_iter=3000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

gridsearch 모듈을 이용하여 위의 ML 모델에 테스트해보고 score, time 등을 뽑아내어, 이 때 가장 좋은 score를 낼 때의 hyperparameter를 기록한다.

- #### 3.2.3 Machine Learning



- ### 3.3. inference_dag



## 4. 