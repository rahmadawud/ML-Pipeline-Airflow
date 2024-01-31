""" Airflow DAG to train and evaluate a traffic signs classification model """


from airflow.decorators import dag, task
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
from model_utils import model_eval, train_model
from data_utils import data_preparation, download_and_unzip


# Define important paths
zip_file_path = "/opt/airflow/data/traffic-signs-data.zip"
raw_data_path = "/opt/airflow/data/raw/"
preprocessed_data_path = "/opt/airflow/data/preprocessed/"
output_path = "/opt/airflow/data/output/"
data_url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"
NUM_EPOCHS = 10


# Define the DAG
@dag("Traffic-Signs-Classification", start_date=days_ago(0), schedule="@monthly", catchup=False)
def ml_pipeline():
    # Download and unzip task
    download_and_unzip_task = PythonOperator(
        task_id="download_and_unzip",
        python_callable=download_and_unzip,
        op_kwargs={
            "url": data_url,
            "zip_file_path": zip_file_path,
            "extract_to_path": raw_data_path,
        },
    )

    # unzip_task = BashOperator(
    #     task_id='unzip_traffic_signs_data',
    #     bash_command='tar -xf opt/airflow/data/traffic-signs-data.zip -C opt/airflow/data/raw/traffic-signs-data',
    # )
    # Data preparation task
    data_prep_task = PythonOperator(
        task_id="data_prep",
        python_callable=data_preparation,
        op_kwargs={
            "input_data_path": raw_data_path,
            "output_data_path": preprocessed_data_path,
        },
    )
    # Model training task
    model_train_task = PythonOperator(
        task_id="model_train",
        python_callable=train_model,
        op_kwargs={"data_path": preprocessed_data_path, "output_path": output_path, "num_epochs": NUM_EPOCHS},
    )
    # Model evaluation task
    model_eval_task = PythonOperator(
        task_id="model_eval",
        python_callable=model_eval,
        op_kwargs={"data_path": preprocessed_data_path, "output_path": output_path},
    )

    # Branching task
    def branch_task(**kwargs):
        if kwargs["ti"].xcom_pull(task_ids="model_eval", key="model_eval_accuracy") > 0.8:
            return "deploy_model"
        else:
            return "do_not_deploy_model"
    
    check_accuracy = BranchPythonOperator(
        task_id="check_accuracy",
        python_callable=branch_task,
        provide_context=True
    )

    # dummy deploy task
    deploy_model = DummyOperator(
        task_id="deploy_model",
    )
    # dummy do not deploy task
    do_not_deploy_model = DummyOperator(
        task_id="do_not_deploy_model",
    )

    download_and_unzip_task >> data_prep_task >> model_train_task >> model_eval_task
    model_eval_task >> check_accuracy
    check_accuracy >> [deploy_model, do_not_deploy_model]



ml_pipeline()
