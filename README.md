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


- ### 3.2. learning_dag


- ### 3.3. inference_dag



## 4. 