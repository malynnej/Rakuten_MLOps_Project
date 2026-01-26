# Repository architecture
- For describing and discussing the repository architecture
- To be removed if not useful anymore

## Sketch of folder architecture
```
Rakuten_MLOps_Project
├── data
│   ├── raw
│   ├── preprocessed
│   └── test_samples      #json files with samples from holdout set
├── models
│   ├── <model_1>
│   └── <model_2>
├── results
│   ├── <model_1>
│   │   ├── evaluation
│   │   └── ... (maybe predictions, etc.)
│   └── <model_2>
├── config
│   ├── paths.yaml
│   └── params.yaml
├── deployments
│   ├── nginx
│   │   └── nginx.conf
│   └── ... (more services)
├── src
│   ├── data
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── core
│   │   │   ├── __init__.py
│   │   │   └── config.py
│   │   ├── services
│   │   │   ├── __init__.py
│   │   │   ├── data_import
│   │   │   │   ├── __init__.py
│   │   │   │   └── import_raw_data.py
│   │   │   └── preprocess
│   │   │       ├── __init__.py
│   │   │       ├── text_cleaning.py
│   │   │       ├── text_outliers.py
│   │   │       └── text_preparation_pipeline.py
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   ├── train_model
│   │   ├── api.py
│   │   ├── core
│   │   │   ├── __init__.py
│   │   │   └── config.py
│   │   ├── services
│   │   │   ├── __init__.py
│   │   │   └── train_model_text.py
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   ├── evaluate_model
│   │   ├── api.py
│   │   ├── core
│   │   │   ├── __init__.py
│   │   │   └── config.py
│   │   ├── services
│   │   │   ├── __init__.py
│   │   │   └── evaluate_text.py
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   ├── predict
│   │   ├── api.py
│   │   ├── core
│   │   │   ├── __init__.py
│   │   │   └── config.py
│   │   ├── services
│   │   │   ├── __init__.py
│   │   │   ├── text_cleaning.py
│   │   │   ├── text_outliers.py
│   │   │   └── text_preparation_predict.py
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   └── ... (more services)
├── tests
│   ├── test_predict_api
│   ├── test_data_api
│   ├── ... (more tests)
│   ├── Dockerfile
│   └── pyproject.toml
├── notebooks
└── logs
Makefile
docker-compose.yml
pyproject.toml
uv.lock
.gitignore
```
generated with [ASCII Text Tree Generator](https://www.text-tree-generator.com/)

