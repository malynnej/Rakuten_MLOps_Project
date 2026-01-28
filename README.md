Project Name
==============================

Rakuten Challenge MLOps


Project Organization (`TO BE UPDATED`)
------------

    ├── LICENSE
    │
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── logs               <- Logs from training and predicting
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks for testing purposes
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── api            <- API file(s)
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── check_structure.py    <- checks if a file or folder exists
    │   │   ├── import_raw_data.py    <- imports raw data
    │   │   │   
    │   │   └── raw        <- raw data files
    │   │       ├── image_train    <- Where you put the images of the train set
    │   │       ├── image_test     <- Where you put the images of the test set
    │   │       ├── X_train_update.csv     <- The text train csv file with the columns designation, description, productid imageid
    │   │       ├── Y_train_update.csv     <- The text train csv file with the target classes
    │   │       ├── X_test_update.csv     <- The text test csv file with the columns designation, description, productid imageid (not used currently as there is no corresponding y_test file provided)
    │   │   │   
    │   │   └── preprocessed        <- preprocessed data files
    │   │   │   
    │   │   └── processed        <- processed encoded test dataset
    │   │   │   
    │   │   └── results        <- evaluation metrics (confusion matrix and classifation report)
    │   │
    │   ├── features       <- Preprocessing scripts to turn raw data into features for modeling with preprocessing_pipeline.py as main pipeline
    │   │
    │   ├── models
    │   │   ├── evaluate_text.py    <- evaluation of trained text model (outputs confusion matrix and classification report )                
    │   │   ├── predict_text.py    <- prediction with trained text model 
    │   │   └── train_model_text.py   <- text training file


--------

Setup Local Repository (`TO BE UPDATED`)
------------

> `git clone https://github.com/malynnej/Rakuten_MLOps_Project.git ` <- clones remote repository

Once you have downloaded and connected the github repo, open the folder in your command tool and follow those instructions :

ATTENTION: Please make sure to be at the project root before executing commands (cd to MLOps_classification_e-commerce)!

Set up of virtual environment
> `curl -LsSf https://astral.sh/uv/install.sh | sh`    <- It will install uv on MAC, if needed, check with 'export PATH="$HOME/.local/bin:$PATH" ' that uv is correctly located, optionally check with 'uv --version' that the uv is installed correctly

> `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" ` <- install uv on Windows/Powershell, optionally check with 'uv --version' that the uv is installed correctly

> `uv sync --python 3.11`   <- sync environment, installs python and packages, creates .venv in directory

> `source .venv/bin/activate`            <- activate virtual environment on Mac

> `.venv\Scripts\activate`            <- activate virtual environment on Windows

> `uv init`      <- initialize project

> `uv sync`   <-  synchronize environment

* Install GNU Make (commands from ChatGPT, not tested!)
  - For Linux/Ubuntu, run `sudo apt-get install build-essential` in the terminal.
  - For Windows, download the installer from the GnuWin32 website.
  - For Mac, ensure Xcode is installed and then run `xcode-select --install` in the terminal.

* Install Docker
  - [Docker Engine](https://docs.docker.com/engine/install/)
  - [Docker Desktop](https://docs.docker.com/desktop/)


Install new libraries (`TO BE UPDATED`)
------------
If you work on this repository and install/add new libraries, please follow this workflow:

> `uv add <library_name>`   <-  add libraries to pyproject.toml
> `uv sync`     <- install/update libraries according to pyproject.toml
> `uv export --no-hashes --format requirements-txt > requirements.txt`  <- creates/updates requirements.txt

SERVICE PIPELINE (UPDATED)
------------

DVC is used  to version control data, models, and evaluation output
------------
Tracked via DVC:
data/raw/
data/preprocessed/
models/bert-rakuten-final
metrics/metrics.json
> `uv run dvc repro`  <- reproduce the full pipeline

DVC artifacts are stored in a shared DAGsHub remote to ensure reproducibility across the team.
> `uv run dvc pull`  <- To pull tracked data and models

Import raw data
------------
The raw data is not tracked by Github due to its size (added to .gitignore), so please import it once in your local repository:

Import text data
> `uv run python -m services.data_import.import_raw_data`  <- imports raw data, execute from src/data folder (cd command to src/data folder)

Preprocessing
------------
> `uv run python -m services.preprocess.text_preparation_pipeline`  <- run preprocessing , execute from src/data folder (cd command to src/data folder)

Training
------------
> `uv run python -m services.train_model_text`    <- run training, execute from src/train_model folder (cd command to src/train_model folder)

Evaluation
------------
> `uv run python -m services.evaluate_text`  <- run evaluation (confusion matrix + class report), execute from src/evaluate_model folder (cd command to src/evaluate_model folder)

Prediction
------------
> `uv run python -m services.predict_text --text "Bloc skimmer PVC sans eclairage;<p>Facile à installer : aucune découpe de paroi ni de liner. <br />Se fixe directement sur la margelle. Adaptateur balai<br />. Livré avec panier de skimmer. </p><br /><ul><li><br /></li><li>Dimensions : 61 x 51 cm</li><li><br /></li><li>Inclus : Skimmer buse de refoulement</li><li><br /></li></ul>" --probabilities --top_k 3`  <- run prediction test, execute from src/predict folder (cd command to src/predict folder)

> `uv run python -m services.predict_text --designation "Bloc skimmer PVC sans eclairage" --description "<p>Facile à installer : aucune découpe de paroi ni de liner. <br />Se fixe directement sur la margelle. Adaptateur balai<br />. Livré avec panier de skimmer. </p><br /><ul><li><br /></li><li>Dimensions : 61 x 51 cm</li><li><br /></li><li>Inclus : Skimmer buse de refoulement</li><li><br /></li></ul>" --probabilities --top_k 5`  <- run prediction test (designation + description), execute from src/predict folder (cd command to src/predict folder)

Eperiment Tracking (MLflow)
------------
> `uv run mlflow ui --host 0.0.0.0 --port 5000`  <- prepares the tracking infrastructure; experiment logging will be extended in follow-up work


API 
------------

Start API from each service (currently tested from each service folder)

`uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload`  <- predict 

`uv run uvicorn api:app --host 0.0.0.0 --port 8001 --reload`  <- data

`uv run uvicorn api:app --host 0.0.0.0 --port 8002 --reload`  <- training

`uv run uvicorn api:app --host 0.0.0.0 --port 8004 --reload`  <- evaluation

Checks

`curl http://localhost:8001/health` <- health check, replace with respective port of service

`curl http://localhost:8001/status` <- status check, replace with respective port of service

`curl http://localhost:8001/results/latest` <- latest results, replace with respective port of service

Data Service (most important)

`curl -X POST http://localhost:8001/import/raw` <- import data

`curl -X POST http://localhost:8001/preprocess/from-raw -H "Content-Type: application/json" -d '{"combine_existing_data": false,"save_holdout": true}'` <- preprocess raw data

Training (most important)

`curl -X POST http://localhost:8002/train -H "Content-Type: application/json" -d '{"retrain": false,"model_name": "bert-rakuten-v1.0.0"}'` <- initial training


Evaluation (most important)

`curl -X POST http://localhost:8004/evaluate -H "Content-Type: application/json" -d '{"batch_size": 32,"model_name": "bert-rakuten-final"}'` <- run evaluation

Prediction (most important)

`curl -X POST http://localhost:8000/predict/text -H "Content-Type: application/json" -d '{"text": "Bloc skimmer PVC sans eclairage;<p>Facile à installer : aucune découpe de paroi ni de liner. <br />Se fixe directement sur la margelle. Adaptateur balai<br />. Livré avec panier de skimmer. </p><br /><ul><li><br /></li><li>Dimensions : 61 x 51 cm</li><li><br /></li><li>Inclus : Skimmer buse de refoulement</li><li><br /></li></ul>", "return_probabilities": true,"top_k": 3}'` <- single prediction (text)

`curl -X POST http://localhost:8000/predict/product -H "Content-Type: application/json" -d '{"designation": "Bloc skimmer PVC sans eclairage","description": "<p>Facile à installer : aucune découpe de paroi ni de liner. <br />Se fixe directement sur la margelle. Adaptateur balai<br />. Livré avec panier de skimmer. </p><br /><ul><li><br /></li><li>Dimensions : 61 x 51 cm</li><li><br /></li><li>Inclus : Skimmer buse de refoulement</li><li><br /></li></ul>","return_probabilities": true,"top_k": 3}'` <- single prediction (designation + description)

Docs accessible (if API is running, replace with respective port of service):
http://127.0.0.1:8000/docs


Start services
--------------
* Start / stop all services with make
  ```
  make run_apis
  make stop_apis
  ```
