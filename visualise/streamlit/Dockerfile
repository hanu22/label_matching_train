FROM python:3.9-slim-buster

COPY requirements.in /app/
WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir pip-tools \
    && pip-compile --upgrade --generate-hashes requirements.in \
    && pip-sync requirements.txt

COPY .streamlit ./.streamlit/
COPY data ./data/
COPY images ./images/
COPY scripts ./scripts/
COPY app.py /app/

EXPOSE 8080 2222
CMD streamlit run app.py
