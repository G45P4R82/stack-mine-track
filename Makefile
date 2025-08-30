build:
	docker build -t stack-mine-track:latest .
run:
	docker run -d --name stack-mine-track stack-mine-track:latest
stop:
	docker stop stack-mine-track && docker rm stack-mine-track	
	
logs:
	docker logs -f stack-mine-track

publish:
	docker tag stack-mine-track:latest srmarinho/stack-mine-track:latest
	docker push srmarinho/stack-mine-track:latest	

login:
	docker login


help:
	@echo "Makefile commands:"
	@echo "  build  - Build the Docker image"
	@echo "  run    - Run the Docker container"
	@echo "  stop   - Stop and remove the Docker container"
	@echo "  logs   - Follow the logs of the Docker container"
	@echo "  help   - Show this help message"