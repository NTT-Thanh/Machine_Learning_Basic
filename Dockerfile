FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train model when building the image if it does not exist.
RUN python train.py

EXPOSE 5000

CMD ["python", "main.py"]
