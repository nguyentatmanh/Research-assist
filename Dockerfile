FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
ARG INSTALL_LOCAL_MODELS=0
RUN if [ "$INSTALL_LOCAL_MODELS" = "1" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install '.[local-models]'; \
    else pip install .; fi \
    && useradd --create-home --uid 10001 researcher \
    && mkdir -p /app/data && chown researcher:researcher /app/data
USER researcher
ENV RA_HOST=0.0.0.0 RA_PORT=8000 RA_DATA_DIR=/app/data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=3)"
CMD ["research-assist", "serve"]
