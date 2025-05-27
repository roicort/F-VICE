# app/Dockerfile

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.uv.tools | bash

COPY . .

RUN uv run

EXPOSE 8888

HEALTHCHECK CMD curl --fail http://localhost:8888/_stcore/health

ENTRYPOINT ["uv","run", "streamlit", "run", "app.py", "--server.port=8888", "--server.address=0.0.0.0"]