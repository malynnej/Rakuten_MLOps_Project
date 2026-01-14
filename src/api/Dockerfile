FROM python:3.11-slim

# install curl for health check command
RUN apt-get update \
 && apt-get install -y curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/api/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api/api.py src/api/api.py

COPY src/models/ src/models/

COPY models/ models/

CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
