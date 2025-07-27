# --- Stage 1: The "Builder" Stage ---
# This stage's only purpose is to install dependencies into a clean virtual environment
FROM python:3.10-slim AS builder

# Set a working directory for our build artifacts
WORKDIR /opt/venv

# First, copy ONLY the requirements file into a temporary location
COPY ./requirements.txt /tmp/requirements.txt

# Create a virtual environment inside the builder stage
RUN python3 -m venv .

# Activate the virtual environment and install dependencies from the copied file
# This command is now a single RUN instruction for better caching
RUN . bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# --- Stage 2: The "Final" Stage ---
# This is the actual image we will run our application from
FROM python:3.10-slim

# Set the final working directory for the application
WORKDIR /code

# Copy ONLY the installed libraries from the builder stage's virtual environment
# This is the key benefit of a multi-stage build
COPY --from=builder /opt/venv /opt/venv

# Copy your application source code into the final image
COPY ./app /code/app

# Set the PATH environment variable so that the container uses the Python
# and packages from the virtual environment we copied over.
ENV PATH="/opt/venv/bin:$PATH"

# Tell Docker that the application will listen on port 8000
EXPOSE 8000

# The command to run when the container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]