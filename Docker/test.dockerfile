FROM python:3.10.13-slim

COPY test.py test.py

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/* 

# install python packages
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# entrypoint
ENTRYPOINT ["python", "-m", "test"]