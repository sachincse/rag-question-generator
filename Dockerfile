# --- Stage 1: The "Builder" Stage ---
FROM python:3.10-slim AS builder
WORKDIR /opt/venv
COPY ./requirements.txt /tmp/requirements.txt
RUN python3 -m venv .
RUN . bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# --- Stage 2: The "Final" Stage ---
FROM python:3.10-slim
WORKDIR /code
COPY --from=builder /opt/venv /opt/venv
COPY ./app /code/app
ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]