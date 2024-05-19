.PHONY: setup-backend setup-frontend run-backend run-frontend install-deps-backend install-deps-frontend run-app setup-app help

# Définir le shell
SHELL := /bin/bash

# Setup de l'environnement virtuel et installation des dépendances pour le backend
setup-backend:
	@echo "Verification de l'environnement virtuel pour le backend..."
	cmd /c "if not exist app\\backend\\venv_backend (cd app\\backend && python -m venv venv_backend)"
	@echo "Installation des dependances pour le backend..."
	$(MAKE) install-deps-backend

# Setup de l'environnement virtuel et installation des dépendances pour le frontend
setup-frontend:
	@echo "Creation de l'environnement virtuel pour le frontend..."
	cmd /c "if not exist app\\frontend\\venv_frontend (app/frontend && python -m venv venv_frontend)"
	@echo "Installation des dépendances pour le frontend..."
	$(MAKE) install-deps-frontend

# Installer les dépendances du backend
install-deps-backend:
	cmd /c "cd app\\backend && venv_backend\\Scripts\\activate && pip install -r requirements.txt"

# Installer les dépendances du frontend
install-deps-frontend:
	cmd /c "cd app\\frontend && venv_frontend\\Scripts\\activate && pip install -r requirements.txt"

# Exécuter le serveur FastAPI
run-backend:
	cmd /c "cd app/backend && venv_backend\\Scripts\\activate && uvicorn main:app --reload"

# Exécuter l'application Streamlit
run-frontend:
	cmd /c "cd app\\frontend && venv_frontend\\Scripts\\activate && streamlit run frontend_main.py"

# Exécuter les deux serveurs en parallèle
run-app:
	start cmd /c "make run-backend"
	start cmd /c "make run-frontend"

setup-app:
	$(MAKE) setup-backend
	$(MAKE) setup-frontend


# Chemin vers l'environnement virtuel Python
VENV_BACKEND := venv_backend
VENV_FRONTEND := venv_frontend

# Spec file pour PyInstallerk
SPEC_FILE_BACKEND := Backend.spec
SPEC_FILE_FRONTEND := frontend.spec

# Dossier de destination pour le build
DIST_PATH := .\\..\\builds

# Version de l'application (peut être dynamiquement mise à jour avec, par exemple, un numéro de build)
VERSION ?= $(shell python -c "import os; print(max([0]+[int(d.split('V')[1]) for d in os.listdir('app/builds') if 'V' in d])+1)")

build-backend: setup-backend
	@echo "Building Backend with PyInstaller..."
	@echo "Version: $(VERSION)"
	@echo "Building in $(DIST_PATH)/V$(VERSION)"
	cmd /c "cd app\\backend && $(VENV_BACKEND)\\Scripts\\activate && $(VENV_BACKEND)\\Scripts\\pyinstaller --distpath $(DIST_PATH)\\V$(VERSION) --workpath .\\build\\V$(VERSION) --clean --noconfirm --log-level=WARN $(SPEC_FILE_BACKEND)"
	@echo "Build completed"

build-frontend:
	@echo "Building Frontend with PyInstaller..."
	@echo "Version: $(VERSION)"
	@echo "Building in $(DIST_PATH)/V$(VERSION)"
	cmd /c "cd app\\frontend && $(VENV_FRONTEND)\\Scripts\\activate && $(VENV_FRONTEND)\\Scripts\\pyinstaller --distpath $(DIST_PATH)\\V$(VERSION) --workpath .\\build\\V$(VERSION) --clean --noconfirm --log-level=WARN $(SPEC_FILE_FRONTEND)"
	@echo "Build completed"


help:
	@echo "setup-backend: Setup de l'environnement virtuel et installation des dépendances pour le backend"
	@echo "setup-frontend: Setup de l'environnement virtuel et installation des dépendances pour le frontend"
	@echo "run-backend: Exécuter le serveur FastAPI"
	@echo "run-frontend: Exécuter l'application Streamlit"
	@echo "run-app: Exécuter les deux serveurs en parallèle"
	@echo "install-deps-backend: Installer les dépendances du backend"
	@echo "install-deps-frontend: Installer les dépendances du frontend"
	@echo "help: Afficher ce message d'aide"
