FROM python:3.11-slim

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
