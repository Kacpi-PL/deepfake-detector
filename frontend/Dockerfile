FROM python:3.11-slim
WORKDIR /app
COPY requirements-front.txt .
RUN pip install --no-cache-dir -r requirements-front.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app_frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]