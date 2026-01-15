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

Setup Local Repository
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


Install new libraries
------------
If you work on this repository and install/add new libraries, please follow this workflow:

> `uv add <library_name>`   <-  add libraries to pyproject.toml
> `uv sync`     <- install/update libraries according to pyproject.toml
> `uv export --no-hashes --format requirements-txt > requirements.txt`  <- creates/updates requirements.txt

Import raw data
------------
The raw data is not tracked by Github due to its size (added to .gitignore), so please import it once in your local repository:

Import images
> Upload the image data folder set directly on local from https://challengedata.ens.fr/participants/challenges/35/, you should save the folders image_train and image_test respecting the following structure

    ├── data
    │   └── raw           
    |   |  ├── image_train 
    |   |  ├── image_test 

Import text data
> `python data/import_raw_data.py`  <- imports raw data, execute from src folder (cd command to src folder)

Preprocessing
------------
> `uv run python -m src.features.preprocessing_pipeline`  <- run preprocessing (due to path issues use uv run instead of just python)

> `uv run python -m src.models.train_model_text`    <- run training

Evaluation
------------
> `uv run python -m src.models.evaluate_text --model_path ./models/bert-rakuten-final --dataset_path ./src/data/processed/test_dataset --output_dir ./src/data/results/evaluation`  <- run evaluation (confusion matrix + class report)

API
------------
`uvicorn src.api.api:app --reload`   <- run API (stop with CTRL + C)

Docs accessible (if API is running):
http://127.0.0.1:8000/docs


Start services
--------------
* Start / stop all services with make
  > `make run_apis`
  > `make stop_apis`
