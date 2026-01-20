FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /tests

# set up virtual environment
ENV VENV_PATH=/app/venv
RUN uv venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# install dependencies from lock file
COPY pylock.toml .
RUN uv pip sync pylock.toml

COPY test_api_predict.py .

CMD ["python", "test_api_predict.py"]
