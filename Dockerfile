# Python Image
FROM python:3.12-slim

# Copy uv into the container
COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /uvx /bin/

# Install system dependencies for lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Python container settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_DEV=1

# Copy dependency files and install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

COPY . .

# Install the project into the container environment
RUN uv sync --locked --no-editable

# Run as a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]
