
# Data generation
data:
	python -m data.make_dataset
data_build:
	docker build -f docker-images/data.dockerfile . -t data:latest
data_run:
	docker run data:latest


# Training
train:
	python -m models.train
train_build:
	docker build -f docker-images/trainer.dockerfile . -t trainer:latest
train_run:
	docker run trainer:latest


# Evaluation
evaluate:
	python -m models.evaluate
evaluate_build:
	docker build -f docker-images/evaluator.dockerfile . -t evaluator:latest
evaluate_run:
	docker run evaluator:latest


# GPU testing
gpu_build:
	docker build -f docker-images/gpu.dockerfile . -t gpu:latest
gpu_run:
	docker run --gpus all gpu:latest
