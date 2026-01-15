# Repository architecture
- For describing and discussing the repository architecture
- To be removed if not useful anymore

## Sketch of folder architecture
```
Rakuten_MLOps_Project
├── data
│   ├── raw
│   ├── processed
│   └── ...
├── models
│   ├── <model_1>
│   └── <model_2>
├── results
│   ├── <model_1>
│   │   ├── evaluation
│   │   └── ... (maybe predictions, etc.)
│   └── <model_2>
├── src
│   ├── data
│   │   ├── import_raw.py
│   │   ├── preprocess.py
│   │   ├── api.py
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   ├── train_model
│   │   ├── train_model.py
│   │   ├── api.py
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   ├── evaluate_model
│   │   ├── evaluate_model.py
│   │   ├── api.py
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   ├── predict
│   │   ├── predict.py
│   │   ├── api.py
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   ├── nginx
│   │   ├── Dockerfile
│   │   └── nginx.conf
│   └── ... (more services)
├── tests
│   ├── test_predict_api
│   ├── test_data_api
│   ├── ... (more tests)
│   ├── Dockerfile
│   └── pyproject.toml
├── notebooks
└── logs
docker-compose.yml
pyproject.toml
uv.lock
.gitignore
```
