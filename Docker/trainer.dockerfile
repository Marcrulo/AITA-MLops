# base image
FROM nvcr.io/nvidia/pytorch:22.07-py3

# move files to container
COPY requirements.txt requirements.txt
COPY data/make_dataset.py data/make_dataset.py
COPY data/tokenized data/tokenized
COPY models/ models/
COPY configs/ configs/

# install python packages
RUN pip install -r requirements.txt --no-cache-dir

# entrypoint
ENTRYPOINT ["python", "-m", "models.train"]