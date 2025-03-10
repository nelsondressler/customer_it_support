# Use an official Python runtime as a parent image
FROM python:3.11.6-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential

# Create working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Poetry
RUN pip install poetry

# Install project dependencies
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Expose the port (for FastAPI, default 8000)
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["poetry", "run", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]