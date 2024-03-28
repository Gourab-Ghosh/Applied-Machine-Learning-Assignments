FROM python:3-slim-bullseye

WORKDIR /app

COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8576

CMD ["python3", "app.py"]