"""Example DAG demonstrating the usage of the PythonOperator."""
import time
from pprint import pprint

from airflow import DAG
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator
from airflow.utils.dates import days_ago

molding_brand_name = ['WooJin', 'DongShin']
woojin_factory_name = ['NewSeoGwang', 'saasdfq']
dongshin_factory_name = ['teng', 'sdfsdf1','333' ]


args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='python_operator',
    default_args=args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=['example'],
) as dag:

    # [START howto_operator_python]
    def print_context(ds, **kwargs):
        """Print the Airflow context and ds variable from the context."""
        pprint(kwargs)
        print(ds)
        return 'Whatever you return gets printed in the logs'

    run_this = PythonOperator(
        task_id='print_the_context',
        python_callable=print_context,
    )
    # [END howto_operator_python]

    # [START howto_operator_python_kwargs]
    
    def teng_func():
        print('congratulations!!')
        
    def NewSeoGwang_func():
        print('congratul!')
    
    def my_sleeping_function(**kwargs):
        """This is a function that will run within the DAG execution"""
        brand_name=kwargs['brand_name']
        factory_name=kwargs['factory_name']
        func=f'{factory_name}_func'
        try:
            eval(func)()
        except Exception as e:
            print(e)
            print(func)
    
            

    # Generate 5 sleeping tasks, sleeping from 0.0 to 0.4 seconds respectively
    for i in molding_brand_name:
        brand_name=i.lower()
        fact=f'{brand_name}_factory_name'
        fact_list=eval(fact)
        for j in fact_list:
            task = PythonOperator(
                task_id='sleep_for_' + i + '_' + j,
                python_callable=my_sleeping_function,
                op_kwargs={'brand_name': brand_name, 'factory_name':j},
            )
            run_this >> task
    # [END howto_operator_python_kwargs]

    # [START howto_operator_python_venv]
    def callable_virtualenv():
        """
        Example function that will be performed in a virtual environment.
        Importing at the module level ensures that it will not attempt to import the
        library before it is installed.
        """
        from time import sleep

        from colorama import Back, Fore, Style

        print(Fore.RED + 'some red text')
        print(Back.GREEN + 'and with a green background')
        print(Style.DIM + 'and in dim text')
        print(Style.RESET_ALL)
        for _ in range(10):
            print(Style.DIM + 'Please wait...', flush=True)
            sleep(10)
        print('Finished')

    virtualenv_task = PythonVirtualenvOperator(
        task_id="virtualenv_python",
        python_callable=callable_virtualenv,
        requirements=["colorama==0.4.0"],
        system_site_packages=False,
    )
