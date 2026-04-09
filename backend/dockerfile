FROM python:3.12-slim

# instalar libgomp
RUN apt-get update && apt-get install -y libgomp1

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend .

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]