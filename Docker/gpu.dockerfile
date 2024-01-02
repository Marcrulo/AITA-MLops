# base image
FROM nvcr.io/nvidia/pytorch:22.07-py3

COPY requirements.txt requirements.txt
COPY gpu_test.py gpu_test.py
COPY config/ config/

# install python packages
RUN pip install -r requirements.txt --no-cache-dir

# entrypoint
ENTRYPOINT ["python", "-u", "gpu_test.py"]