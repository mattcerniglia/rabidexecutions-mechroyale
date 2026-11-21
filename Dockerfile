FROM python:3.13-slim
WORKDIR /bot
RUN pip install --no-cache-dir numpy
COPY . .
ENTRYPOINT ["python3", "-u", "/bot/bot_v2.py"]
