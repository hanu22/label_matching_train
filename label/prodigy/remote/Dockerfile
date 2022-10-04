FROM python:3.9-slim-buster

COPY requirements.in /prodigy/
WORKDIR /prodigy

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir pip-tools \
    && pip-compile --upgrade --generate-hashes requirements.in \
    && pip-sync requirements.txt

COPY wheel/prodigy-xx.whl ./wheel/
RUN pip install wheel/prodigy-xx.whl \
    && rm -rf wheel/prodigy-xx.whl
RUN python -m spacy download en_core_web_xx

COPY prodigy.json .
COPY ./data ./data/

ENV PRODIGY_HOME /prodigy
ENV PRODIGY_LOGGING "verbose"
ENV PRODIGY_ALLOWED_SESSIONS "user1,user2"
ENV PRODIGY_BASIC_AUTH_USER "PRODIGY"
ENV PRODIGY_BASIC_AUTH_PASS "password"

EXPOSE 80 2222

CMD python -m prodigy component repo-name model ./data/dataset.jsonl --label LABEL1,LABEL2,LABEL3 ./data/patterns.jsonl
