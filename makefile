# Set default goal to 'help' for Makefile
.DEFAULT_GOAL := help

# Define environment setup
init:
	docker compose up airflow-init
# Build and start Airflow with custom images
start:
	docker compose up --build

# Stop and clean up environment
clean:
	docker compose down --volumes --rmi all

# Display help message
help:
	@echo "Usage:"
	@echo "  make init    - Prepare the environment for the first Airflow run"
	@echo "  make start   - Start Airflow and build custom images"
	@echo "  make clean   - Stop and clean up the environment"
	@echo "  make help    - Display this help message"
