FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN adduser --disabled-password --gecos "" appuser

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

# 注意：--rebuild-only 写入 /data/index 需宿主机目录对容器用户可写
USER appuser

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.port=8501"]
