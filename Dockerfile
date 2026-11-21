FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
WORKDIR /bot
RUN pip install --no-cache-dir numpy==2.5.5
COPY . .
ENTRYPOINT ["python3", "-u", "/bot/bot_v2.py"]
