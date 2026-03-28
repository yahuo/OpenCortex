FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8501 \
    API_PORT=8502

RUN adduser --disabled-password --gecos "" appuser

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

# 注意：--rebuild-only 写入 /data/index 需宿主机目录对容器用户可写
USER appuser

EXPOSE 8501 8502

CMD ["python", "start.py", "--skip-rebuild"]
