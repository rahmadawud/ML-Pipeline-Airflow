# Airflow ML Pipeline for Traffic Signs Classification

## Overview

## Usage

To use this pipeline for local development, follow the steps below:

1. Ensure that your Docker Engine has sufficient memory allocated, as running the pipeline may require more memory in certain cases.

2. Be in the project directory

3. Before the first Airflow run, prepare the environment by executing the following steps:

   - If you are working on Linux, specify the AIRFLOW_UID by running the command:

   ```bash
   echo -e "AIRFLOW_UID=$(id -u)" > .env
   ```

   - Perform the database migration and create the initial user account by running the command:

   ```bash
   docker compose up airflow-init
   ```

   The created user account will have the login `airflow` and the password `airflow`.

4. Start Airflow and build custom images to run tasks in Docker-containers:

   ```bash
   docker compose up --build
   ```

5. Access the Airflow web interface in your browser at http://localhost:8080.

6. Trigger the DAG to initiate the pipeline execution.

7. When you are finished working and want to clean up your environment, run:

   ```bash
   docker compose down --volumes --rmi all
   ```
