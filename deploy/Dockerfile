FROM mcr.microsoft.com/azure-functions/python:3.0-python3.9-slim

ENV AzureWebJobsScriptRoot=/home/site/wwwroot
ENV AzureFunctionsJobHost__Logging__Console__IsEnabled=true

WORKDIR /home/site/wwwroot
COPY requirements.in .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir pip-tools \
    && pip-compile --upgrade --generate-hashes requirements.in \
    && pip-sync requirements.txt

COPY model ./model/
RUN pip install model/xxx-py3-none-any.whl \
    && rm -rf model/xxx-py3-none-any.whl

COPY data ./data/
COPY scripts ./scripts/
COPY HttpTrigger ./HttpTrigger/

EXPOSE 80 2222
