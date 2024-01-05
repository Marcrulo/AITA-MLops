# base image
FROM nvcr.io/nvidia/pytorch:22.07-py3

# install python
# RUN apt update && \
#     apt install --no-install-recommends -y build-essential gcc && \
#     apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY gpu_test.py gpu_test.py
COPY configs/ configs/

# install python packages
RUN pip install -r requirements.txt --no-cache-dir

# entrypoint
ENTRYPOINT ["python", "-u", "gpu_test.py"]