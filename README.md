# Machine Learning Pipeline with Apache Airflow

This project implements a machine learning pipeline using Apache Airflow for orchestration and Docker Compose for managing the underlying infrastructure. The pipeline includes tasks for downloading and preprocessing a traffic signs dataset, training a machine learning model, evaluating its performance, and deciding to automatically deploy depending on a threshold.

An accompanying notebook is available under `notebooks/german_traffic_signs_classification.ipynb`

## Usage

To use this pipeline for local development, follow the steps below:

1. Ensure that your Docker Engine has sufficient memory allocated, as running the pipeline may require more memory in certain cases.

2. Be in the project directory.

3. If you are working on Linux, set the AIRFLOW_UID by running the command:
     ```bash
     echo -e "AIRFLOW_UID=$(id -u)" > .env
     ```

4. **Using Docker Compose Directly:**

   - Perform the database migration and create the initial user account with the following command:
     ```bash
     docker compose up airflow-init
     ```
     The created user account will have the login `airflow` and the password `airflow`.
   - Start Airflow and build custom images to run tasks in Docker containers:
     ```bash
     docker compose up --build
     ```

5. **Using Makefile (Alternative):**

   - If you prefer using the Makefile for simplified commands:
     - Run `make init` to set the AIRFLOW_UID and perform the initial database setup.
     - Run `make start` to start Airflow and build custom images.

6. Access the Airflow web interface in your browser at http://localhost:8080. 
  - Username and Password is by default 'airflow'

7. Trigger the Directed Acyclic Graph (DAG) to initiate the pipeline execution.

8. When you have completed your work and want to clean up your environment, run:

   ```bash
   docker compose down --volumes --rmi all

   ```

   or alternatively

   ```bash
       make clean
   ```
