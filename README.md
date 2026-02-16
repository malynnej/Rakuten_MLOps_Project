Rakuten Challenge MLOps
=======================

Repository Structure
--------------------

    Rakuten_MLOps_Project
    ├── data
    │   ├── raw
    │   ├── preprocessed
    │   └── test_samples

-> Data for model training and evaluation

    ├── models

-> Trained models (including label encoder and mappings)

    ├── results

-> Model evaluation results

    ├── config

-> Configuration files needed across multiple services

    ├── src
    │   ├── data
    │   │   ├── core
    │   │   ├── services
    │   │   │   ├── data_import
    │   │   │   └── preprocess
    │   │   ├── Dockerfile
    │   │   └── pyproject.toml
    │   ├── train_model
    │   │   ├── core
    │   │   ├── services
    │   │   ├── Dockerfile
    │   │   └── pyproject.toml
    │   ├── evaluate_model
    │   │   ├── core
    │   │   ├── services
    │   │   ├── Dockerfile
    │   │   └── pyproject.toml
    │   └── predict
    │       ├── core
    │       ├── services
    │       ├── Dockerfile
    │       └── pyproject.toml

-> Core functionality (data handling, model training, evaluation and prediction) 
as individual services and independent sub-projects

    ├── deployments
    │   ├── nginx
    │   ├── prometheus
    │   └── grafana

-> Additional deployments configuration
  
    ├── tests

-> Automated tests

    ├── .dvc

-> DVC config

    ├── .github/workflows

-> Github actions (currently Ruff linting and formatting implemented)

    ├── scripts
    ├── notebooks

-> Development helper scripts and testing notebooks

    ├── streamlit

-> Streamlit presentation of the project

    └── logs

-> Output directory of log files

    Makefile
    docker-compose.yml
    pyproject.toml
    .gitignore

-> Overall project configuration files


Setup / Installation
--------------------

### Docker

See Docker installation instructions for your system
- [Docker Engine](https://docs.docker.com/engine/install/)
- [Docker Desktop](https://docs.docker.com/desktop/)


### GNU Make

Probably some of the following options could work for you
- For Linux, run 
  `sudo apt-get install build-essential`
- For Mac, ensure Xcode is installed and then run 
  `xcode-select --install` 
- For Windows, download the installer from the GnuWin32 website.


### UV

Following the [official installation instructions](
  https://docs.astral.sh/uv/getting-started/installation/):
- Linux / macOS: 
  `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows: 
  `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`


### DVC / DagsHub

To access data tracked by DVC, access to the remote storage on DagsHub has to 
be configured.

* Configure up DagsHub credentials (locally)

      dvc remote modify origin --local auth basic 
      dvc remote modify origin --local user <username>
      dvc remote modify origin --local password <token>

* Create `.env` file with the following information 
  (adapt for using remote MLflow on DagsHub)

      # DVC Configuration
      DVC_HTTP_USER=<username>
      DVC_HTTP_PASSWORD=<token>
      # Mlflow Configuration
      #MLFLOW_TRACKING_URI=https://dagshub.com/jenny-lam/Rakuten_MLOps_Project.mlflow
      MLFLOW_TRACKING_URI=http://mlflow:5000
      MLFLOW_TRACKING_USERNAME=<username>
      MLFLOW_TRACKING_PASSWORD=<token>


Run the project
---------------
* First, pull data from DVC/Dagshub:

      dvc pull

All necessary components of the projects are running in Docker containers.

* Start all services, including API tests by setting up all Docker containers:

      make run_apis
  
  If separate build / start up steps are preferred:

      docker compose build
      docker compose up

nginx serves as a reverse proxy for the API, so the API docs and following services can be accessed via:

* Monitoring with Prometheus/Grafana: https://localhost:8443/grafana/ 
  - Login Grafana - username: admin, pw: admin
  - Login nginx - username: admin1, pw: admin1

* MLFlow UI: http://localhost:5000

* Data API: https://localhost:8443/data/docs

* Train API: https://localhost:8443/train/docs

* Predict API: https://localhost:8443/predict/docs

* Evaluate API: https://localhost:8443/evaluate/docs

Basic user management is set up with nginx, having following roles (username = password):

* credentials for the 3 user classes are hard-coded (generated with `/scripts/generate_nginx_htpasswd.sh`)
  - users (user1, user2) have access to endpoint /predict/
  - devs (dev1, dev2) have access to endpoints /data/, /train/, /evaluate/
  - admins (admin1) have access to endpoint /nginx_status/

Examples of using the API endpoints via command:

* Preprocessing raw data (text preprocessing, label encoding, data split)
  - combine_existing_data: false (default) - processes only raw data, true - combines with existing preprocessed data
  - save_holdout: true (default) - saves holdout set from raw data for testing purposes

        curl -k https://localhost:8443/data/preprocess/from-raw \
          -u dev1:dev1 \
          -X POST -H "Content-Type: application/json" \
          -d '{"combine_existing_data": false,"save_holdout": true}'

* Train BERT-based text model:
  - retrain: false (default) - activates initial model training, true - uses existing trained model for fine-tuning only
  - model_name: name of the model

        curl -k https://localhost:8443/train/train_model \
          -u dev1:dev1 \
          -X POST -H "Content-Type: application/json" \
          -d '{"retrain": false,"model_name": "bert-rakuten-final"}'

* Evaluate text model:
  - batch_size: 32 (default) - batch size for evaluation
  - model_name: name of the model to evaluate   

        curl -k https://localhost:8443/evaluate/evaluate_model \
          -u dev1:dev1 \
          -X POST -H "Content-Type: application/json" \
          -d '{"batch_size": 32,"model_name": "bert-rakuten-final"}'   

* Predict text:
  - text: text input for prediction
  - return_probabilities: true (default) - return probabilities of prediction
  - top_k: 3 - return top k predicted classes

        curl -k https://localhost:8443/predict/predict_text \
          -u dev1:dev1 \
          -X POST -H "Content-Type: application/json" \
          -d '{"text": "Bloc skimmer PVC sans eclairage;<p>Facile à installer : aucune découpe de paroi ni de liner. <br />Se fixe directement sur la margelle. Adaptateur balai<br />. Livré avec panier de skimmer. </p><br /><ul><li><br /></li><li>Dimensions : 61 x 51 cm</li><li><br /></li><li>Inclus : Skimmer buse de refoulement</li><li><br /></li></ul>", "return_probabilities": true,"top_k": 3}'

Further:

* Test results are written to the log files in the `logs` directory, e.g.
  - Tests on predict API: 
    `logs/test_api_predict.log`
  - Tests on nginx: 
    `logs/test_nginx.log`

* Evaluation results are written to the `results` directory, e.g.
  - Classification reports:
    `classification_report_<..>.json`
  - Confusion matrix:
    `confusion_matrix_<..>.png`

* Stop all services with make:

      make stop_apis


Development
-----------

Some helpful commands during development

* If dependencies were added to any `pyproject.toml` files, all affected 
  lockfiles are automatically generated:

      make lockfiles

* Remove all lockfiles (`pylock.toml`):

      make clean_locks

* Remove all log files:

      make clean_logs



### Helper scripts

* `scripts/generate_nginx_certs.sh`: 
  Generate self-signed SSL certificates for nginx reverse proxy

* `scripts/generate_nginx_htpasswd.sh`: 
  Generate user / password files for nginx reverse proxy


Working in Local Environment
----------------------------

Although all components of the project can and should be run as microservices 
in Docker containers, most parts can also be executed locally. This might be 
helpful during development.

### DVC 

Data versioning of data, models, and evaluation output.

* Tracked via DVC:

      data/raw/
      data/preprocessed/
      models/bert-rakuten-final
      models/label_encoder.pkl
      models/label_mappings.json
      metrics/metrics.json
      results

* To reproduce the full pipeline, run: 

      uv run dvc repro

* DVC artifacts are stored in a shared DAGsHub remote to ensure reproducibility 
  across the team. To pull tracked data and models:

      uv run dvc pull


### Import raw data

The raw data is not tracked by Github due to its size (added to .gitignore), 
so please import it once in your local repository:

Import text data, execute from `src/data` folder:

    uv run python -m services.data_import.import_raw_data


### Preprocessing

Run preprocessing  execute from `src/data` folder

    uv run python -m services.preprocess.text_preparation_pipeline


### Training

Run training, execute from `src/train_model` folder

    uv run python -m services.train_model_text


### Evaluation

Run evaluation (confusion matrix + class report), 
execute from `src/evaluate_model` folder

    uv run python -m services.evaluate_text


### Prediction

Run prediction test, execute from `src/predict` folder

    uv run python -m services.predict_text --text "Bloc skimmer PVC sans eclairage;<p>Facile à installer : aucune découpe de paroi ni de liner. <br />Se fixe directement sur la margelle. Adaptateur balai<br />. Livré avec panier de skimmer. </p><br /><ul><li><br /></li><li>Dimensions : 61 x 51 cm</li><li><br /></li><li>Inclus : Skimmer buse de refoulement</li><li><br /></li></ul>" --probabilities --top_k 3


### Experiment Tracking (MLflow)

Prepares the tracking infrastructure, experiment logging will be extended in 
follow-up work

    uv run mlflow ui --host 0.0.0.0 --port 5000


### API 

Start API for each service (from respective directory)

* Predict

      uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload

* Data

      uv run uvicorn api:app --host 0.0.0.0 --port 8001 --reload

* Training

      uv run uvicorn api:app --host 0.0.0.0 --port 8002 --reload

* Evaluation

      uv run uvicorn api:app --host 0.0.0.0 --port 8004 --reload

* Checks

  - Health check, replace with respective port of service

        curl http://localhost:8001/health
  
  - Status check, replace with respective port of service

        curl http://localhost:8001/status

  - Latest results, replace with respective port of service

        curl http://localhost:8001/results/latest

  - Import data

        curl -X POST http://localhost:8001/import/raw
  
  - Preprocess raw data

        curl -X POST http://localhost:8001/preprocess/from-raw \
          -H "Content-Type: application/json" \
          -d '{"combine_existing_data": false,"save_holdout": true}'

  - Initial training

        curl -X POST http://localhost:8002/train_model \
          -H "Content-Type: application/json" \
          -d '{"retrain": false,"model_name": "bert-rakuten-final"}'

  - Run evaluation

        curl -X POST http://localhost:8004/evaluate_model \
          -H "Content-Type: application/json" \
          -d '{"batch_size": 32,"model_name": "bert-rakuten-final"}'`

  - Single prediction (text)

        curl -X POST http://localhost:8000/predict_text \
          -H "Content-Type: application/json" \
          -d '{"text": "Bloc skimmer PVC sans eclairage;<p>Facile à installer : aucune découpe de paroi ni de liner. <br />Se fixe directement sur la margelle. Adaptateur balai<br />. Livré avec panier de skimmer. </p><br /><ul><li><br /></li><li>Dimensions : 61 x 51 cm</li><li><br /></li><li>Inclus : Skimmer buse de refoulement</li><li><br /></li></ul>", "return_probabilities": true,"top_k": 3}'

  - Docs accessible (if API is running, replace respective port of service):

        http://127.0.0.1:8000/docs


Streamlit
---------

Run the streamlit presentation:

    uv run streamlit run streamlit/1_Project_Overview.py
