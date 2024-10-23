FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 3276
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3276"]