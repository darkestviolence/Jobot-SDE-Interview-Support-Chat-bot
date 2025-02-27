FROM python:3.11.1

ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt --default-timeout=100

COPY . .

EXPOSE 8080

CMD ["python", "-m", "chainlit", "run", "model.py", "--host", "0.0.0.0", "--port", "8080"]
