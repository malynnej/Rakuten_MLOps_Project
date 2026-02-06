FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /tests

# set up virtual environment
ENV VENV_PATH=/app/venv
RUN uv venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# install dependencies from lock file
COPY pylock.toml .
RUN uv pip sync pylock.toml

COPY . .

# set log file path
ARG LOGDIR
ENV LOGFILE=${LOGDIR}/test_nginx.log

CMD pytest | tee ${LOGFILE}
