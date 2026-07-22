# Образ веб-сервиса. Модели/данные/эмбеддинги в образ НЕ кладём (они большие) —
# монтируем томами через docker-compose. Контекст сборки = КОРЕНЬ проекта
# (см. docker-compose.yml: build.context = ..), поэтому пути COPY ниже — от корня.
FROM python:3.11-slim

WORKDIR /app

# Зависимости. torch берём в CPU-сборке, чтобы образ был легче.
COPY Final/requirements-web.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-web.txt

# Код приложения (папка app/ лежит в корне проекта).
COPY app ./app

EXPOSE 8000
# host 0.0.0.0 — чтобы сервис был доступен по сети (жюри в общем Wi-Fi).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
