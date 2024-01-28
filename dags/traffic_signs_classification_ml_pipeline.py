""" Airflow DAG to train and evaluate a traffic signs classification model """


from airflow.decorators import dag
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from model_utils import model_eval, train_model
from data_utils import data_preparation, download_and_unzip


# Define important paths
zip_file_path = "/opt/airflow/data/traffic-signs-data.zip"
raw_data_path = "/opt/airflow/data/raw/"
preprocessed_data_path = "/opt/airflow/data/preprocessed/"
output_path = "/opt/airflow/data/output/"
data_url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"

# Define the DAG
@dag("Traffic-Signs-Classification", start_date=days_ago(0), schedule="@daily", catchup=False)
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
        op_kwargs={"data_path": preprocessed_data_path, "output_path": output_path, "num_epochs": 1},
    )
    # Model evaluation task
    model_eval_task = PythonOperator(
        task_id="model_eval",
        python_callable=model_eval,
        op_kwargs={"data_path": preprocessed_data_path, "output_path": output_path},
    )

    download_and_unzip_task >> data_prep_task >> model_train_task >> model_eval_task


ml_pipeline()
