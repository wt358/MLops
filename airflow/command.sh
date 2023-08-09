#!/bin/sh

BRAND=$1
FUNC=$2

git clone https://github.com/wt358/MLops.git
mkdir py-test
cp -r ./MLops/airflow/src/* ./py-test/
# cp ./airflow-DAGS/pyfile/*.py ./py-test/

if [ -e "./conn.sh" ]; then
    ./conn.sh
fi
python3 ./py-test/${BRAND}/copy_gpu_py.py ${FUNC}

# if [ -e "./disconn.sh" ]; then
#     ./disconn.sh
# fi