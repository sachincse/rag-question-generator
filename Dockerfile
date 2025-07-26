# Start from an official Python base image.
# Using 'slim' results in a smaller final image size.
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the dependencies file to the working directory
COPY ./requirements.txt /code/requirements.txt

# Install the Python dependencies
# --no-cache-dir: Reduces image size by not storing the download cache
# --upgrade: Ensures pip is up to date
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the entire 'app' directory into the container
COPY ./app /code/app

# Expose the port the app runs on
EXPOSE 8000

# The command to run when the container starts.
# It tells uvicorn to run the 'app' instance from the 'app.main' module.
# --host 0.0.0.0 makes the server accessible from outside the container.
# --port 8000 specifies the port to run on.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]