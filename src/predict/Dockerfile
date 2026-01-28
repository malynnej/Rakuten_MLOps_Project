FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# install curl for health check command
RUN apt-get update \
 && apt-get install -y curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# set up virtual environment
ENV VENV_PATH=/app/venv
RUN uv venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# install dependencies from lock file
COPY pylock.toml .
RUN uv pip sync pylock.toml

# copy full application code
COPY . .

# run API application
CMD ["uvicorn", "api:app", \
     "--host", "0.0.0.0", "--port", "8000"]
