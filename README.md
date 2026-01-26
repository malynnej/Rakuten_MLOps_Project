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
> `uv run python -m services.predict_text --text "Pgytech Pour Dji Osmo Pocket 4pcs Nd8 Nd16 Nd32 Nd64 Professional Lens Filter @Doauhao3293-Générique; PGYTECH Pour DJI Osmo Pocket 4PCS ND8 Filtre ND64 ND16 nd32 Professional VERRES Caractéristiques: Parfait pour DJI OSMO caméra de poche (non inclus). Contrôlez toutes les situations avec la Cardan. Extrêmement léger châssis en aluminium aviation CNC. Imperméable à l&#39;eau l&#39;huile épreuve et anti-rayures revêtement durci. verre optique SCHOTT allemand pour répondre aux exigences rigoureuses des photographes professionnels. Nanomètre vide multicouche double face pour protéger votre objectif pour améliorer les effets de la clarté et de couleur. La technologie de revêtement multicouche avec de multiples processus de broyage et de polissage afin d&#39;assurer les exigences les plus élevées de la qualité d&#39;image haute définition. Conforme aux normes environnementales de l&#39;UE ROHS de limite à des substances nocives. Système magnétique installation facile et rapide. N&#39;affecter les performances de l&#39;appareil. Caractéristiques: Marque: PGYTECH Matériel:" --probabilities --top_k 3`  <- run prediction test, execute from src/predict folder (cd command to src/predict folder)

> `uv run python -m services.predict_text --designation "Bloc skimmer PVC sans eclairage" --description "<p>Facile à installer : aucune découpe de paroi ni de liner. <br />Se fixe directement sur la margelle. Adaptateur balai<br />. Livré avec panier de skimmer. </p><br /><ul><li><br /></li><li>Dimensions : 61 x 51 cm</li><li><br /></li><li>Inclus : Skimmer buse de refoulement</li><li><br /></li></ul>" --probabilities --top_k 5`  <- run prediction test (designation + description), execute from src/predict folder (cd command to src/predict folder)


API (`TO BE UPDATED`)
------------
`uvicorn src.api.api:app --reload`   <- run API (stop with CTRL + C)

Docs accessible (if API is running):
http://127.0.0.1:8000/docs


Start services
--------------
* Start / stop all services with make
  ```
  make run_apis
  make stop_apis
  ```
