FROM python:3.11-slim

# install uv from prebuilt image
COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /bin/

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
