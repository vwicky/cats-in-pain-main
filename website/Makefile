.PHONY: launch dev install install-web install-frontend

# Full stack: Postgres check, optional DB create, API + worker + Vite
launch:
	bash launch.sh

dev: launch

install: install-web install-frontend

install-web:
	pip install -r requirements.txt

install-frontend:
	cd frontend && npm install

# First-time setup then launch (Python + npm deps)
up:
	bash launch.sh --install-deps
