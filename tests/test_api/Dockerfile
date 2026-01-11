FROM python:3.11-slim

WORKDIR /work

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY test_api.py .

CMD ["python", "test_api.py"]
