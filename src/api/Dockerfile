FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

COPY ./src/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api/api.py src/api/api.py
COPY src/models/predict_text.py src/models/predict_text.py
COPY src/models/train_model_text.py src/models/train_model_text.py
COPY src/models/evaluate_text.py src/models/evaluate_text.py

COPY models/bert-rakuten-final models/bert-rakuten-final

EXPOSE 8000
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
