# Use Python 3.10.14 image
FROM python:3.10.14-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY faiss_index_dir faiss_index_dir
COPY level2_app.py level2_app.py

# Expose the port for Chainlit
EXPOSE 8000

# Start the application
CMD ["python", "level2_app.py"]
