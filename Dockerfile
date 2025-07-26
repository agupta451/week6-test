FROM python:3.10-slim
WORKDIR /app
COPY app/ /app/
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8200
CMD ["uvicorn", "iris_fastapi:app", "--host", "0.0.0.0", "--port", "8200"]
