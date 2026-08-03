FROM apache/airflow:2.10.4-python3.11

USER root
COPY requirements.txt /requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
USER airflow
RUN pip install --no-cache-dir -r /requirements.txt