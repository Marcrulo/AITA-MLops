
# Data generation
dataset:
	python -m data.make_dataset
data_build:
	docker build -f Docker/data.dockerfile . -t data:latest
data_run:
	docker run data:latest


# Training
train:
	python -m models.train
train_build:
	docker build -f Docker/trainer.dockerfile . -t trainer:latest
train_run:
	docker run --rm --gpus all trainer:latest 


# Evaluation
evaluate:
	python -m models.evaluate
evaluate_build:
	docker build -f Docker/evaluator.dockerfile . -t evaluator:latest
evaluate_run:
	docker run evaluator:latest


# GPU testing
gpu_build:
	docker build -f Docker/gpu.dockerfile . -t gpu:latest
gpu_run:
	docker run --gpus all gpu:latest


# Unittest
train_test:
	python -m unittest unittests/train_test.py