# base image
FROM python:3.10.13-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/* 

ENV PORT 8080
ENV HOST 0.0.0.0

# move files to container
COPY requirements.txt requirements.txt
COPY data/make_dataset_gcp.py data/make_dataset_gcp.py
COPY models/ models/
COPY configs/ configs/

# install python packages
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# entrypoint
ENTRYPOINT ["python", "-m", "data.make_dataset_gcp"]