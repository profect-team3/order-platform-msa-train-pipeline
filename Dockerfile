FROM python:3.13-slim

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY config/ ./config/
COPY dags/ ./dags/
COPY scripts/ ./scripts/

# Install dependencies
RUN uv sync

# Expose port
EXPOSE 8000

# Default command
CMD ["uv", "run", "airflow", "standalone"]
