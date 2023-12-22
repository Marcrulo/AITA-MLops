create_dataset:
	python -m data.make_dataset

train:
	python -m models.train

evaluate:
	python -m models.evaluate

docker_data:
	docker build -f data.dockerfile . -t data:latest

docker_train:
	docker build -f trainer.dockerfile . -t trainer:latest