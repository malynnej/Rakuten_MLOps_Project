# Repository architecture
- For describing and discussing the repository architecture
- To be removed if not useful anymore

## Sketch of folder architecture
```
Rakuten_MLOps_Project
├── data
│   ├── raw
│   ├── preprocessed
│   └── test_samples
├── models
├── results
├── config
├── deployments
│   ├── nginx
│   ├── prometheus
│   └── grafana
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
├── tests
├── scripts
├── notebooks
├── streamlit
└── logs
Makefile
docker-compose.yml
pyproject.toml
uv.lock
.gitignore
```
generated with [ASCII Text Tree Generator](https://www.text-tree-generator.com/)

