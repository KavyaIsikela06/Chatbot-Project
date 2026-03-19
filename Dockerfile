FROM python:3.7-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download nltk punkt during build
RUN python3 -c "import nltk; nltk.download('punkt')"

# Default command
CMD ["python3", "app.py"]
