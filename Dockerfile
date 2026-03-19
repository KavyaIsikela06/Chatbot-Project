FROM python:3.7-slim

WORKDIR /app

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy chatbot code
COPY chatbot.py ./

# Create logs directory
RUN mkdir logs

# Run chatbot
CMD ["python", "chatbot.py"]
