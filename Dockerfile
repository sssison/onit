FROM ubuntu:24.04

WORKDIR /app

# Install Python 3 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Use a venv to avoid PEP 668 externally-managed-environment restriction
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency files first (for layer caching)
COPY pyproject.toml ./

# Copy the rest of the project
COPY . .

# Install the project with all dependencies
RUN pip install --no-cache-dir -e ".[all]"

# Default port for web UI
EXPOSE 9000

# Default entrypoint — interactive terminal mode
ENTRYPOINT ["onit"]
