#!/bin/sh

FUNC=$1

git clone https://github.com/wt358/MLops.git
mkdir py-test
cp ./MLops/airflow/src/*.py ./py-test/
# cp ./airflow-DAGS/pyfile/*.py ./py-test/

python3 ./py-test/copy_gpu_py.py ${FUNC}