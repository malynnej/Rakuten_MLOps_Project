FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Install system dependencies (data service might need more)
RUN apt-get update \
 && apt-get install -y curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set up virtual environment
ENV VENV_PATH=/app/venv
RUN uv venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Install dependencies
COPY pylock.toml .
RUN uv pip sync pylock.toml

# Download NLTK data (data service uses text preprocessing)
ENV NLTK_DATA=/usr/share/nltk_data
RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/share/nltk_data')"

# Copy application code
COPY . .

# Set environment
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8001

# Run API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
