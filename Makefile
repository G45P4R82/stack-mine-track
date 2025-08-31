IMAGE_NAME = stack-mine-track
CONTAINER_NAME = stack-mine-track

# Porta para o viz
VIZ_PORT = 8000

build:
	docker build -t $(IMAGE_NAME):latest .

# Executa kedro run no container
run:
	docker run --rm --name $(CONTAINER_NAME) $(IMAGE_NAME):latest run

# Roda Kedro Viz expondo porta para acessar no navegador
viz:
	docker run --rm -p $(VIZ_PORT):$(VIZ_PORT) --name $(CONTAINER_NAME)-viz $(IMAGE_NAME):latest viz --host 0.0.0.0 --port $(VIZ_PORT)

# Para container (se estiver rodando com -d, que aqui não usamos por padrão)
stop:
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

# Logs do container (não aplicável para modo --rm, útil se rodar em -d)
logs:
	docker logs -f $(CONTAINER_NAME)

publish:
	docker tag $(IMAGE_NAME):latest serverlab.lonk-chinstrap.ts.net/library/$(IMAGE_NAME):latest
	docker push serverlab.lonk-chinstrap.ts.net/library/$(IMAGE_NAME):latest

login:
	docker login serverlab.lonk-chinstrap.ts.net -u admin -p Harbor12345

help:
	@echo "Makefile commands:"
	@echo "  build   - Build the Docker image"
	@echo "  run     - Run kedro pipeline (kedro run)"
	@echo "  viz     - Run kedro-viz UI on port $(VIZ_PORT)"
	@echo "  stop    - Stop and remove the Docker container"
	@echo "  logs    - Follow the logs of the Docker container"
	@echo "  publish - Push the image to Harbor registry"
	@echo "  login   - Login to Harbor registry"
