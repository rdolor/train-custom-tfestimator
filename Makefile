IMAGE_NAME=train_custom_tfestimator:master

build ::
	echo ${IMAGE_NAME}
	docker build -f ./Dockerfile -t ${IMAGE_NAME} .
	docker push ${IMAGE_NAME}

train ::
	python -m src.main
  