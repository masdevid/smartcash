# Makefile for SmartCash

# Variables
PYTHON = python
PIP = pip
PYTEST = python -m pytest
COVERAGE = python -m coverage

# Default target
.PHONY: help
help:
	@echo "\n\033[1mSmartCash Development Commands:\033[0m"
	@echo "\n\033[1mInstallation:\033[0m"
	@echo "  install-dev      Install development dependencies"
	@echo "  install          Install package in development mode"
	@echo "\n\033[1mTesting:\033[0m"
	@echo "  test             Run all tests"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  test-unit        Run unit tests"
	@echo "  test-integration Run integration tests"
	@echo "\n\033[1mLinting & Formatting:\033[0m"
	@echo "  lint             Run flake8 linter"
	@echo "  format           Format code with black and isort"
	@echo "  typecheck        Run mypy type checking"
	@echo "  check-all        Run all checks (lint, format, typecheck, test)"

# Installation
.PHONY: install-dev
install-dev:
	@echo "\n\033[1mInstalling development dependencies...\033[0m"
	$(PIP) install -r requirements-dev.txt

.PHONY: install
install:
	@echo "\n\033[1mInstalling in development mode...\033[0m"
	$(PIP) install -e .

# Testing
.PHONY: test
test:
	@echo "\n\033[1mRunning tests...\033[0m"
	$(PYTEST) tests/

.PHONY: test-coverage
test-coverage:
	@echo "\n\033[1mRunning tests with coverage...\033[0m"
	$(PYTEST) --cov=smartcash --cov-report=term-missing tests/

.PHONY: test-unit
test-unit:
	@echo "\n\033[1mRunning unit tests...\033[0m"
	$(PYTEST) tests/unit/

.PHONY: test-integration
test-integration:
	@echo "\n\033[1mRunning integration tests...\033[0m"
	$(PYTEST) tests/integration/

# Linting & Formatting
.PHONY: lint
lint:
	@echo "\n\033[1mRunning flake8...\033[0m"
	flake8 smartcash/

.PHONY: format
format:
	@echo "\n\033[1mFormatting code with black...\033[0m"
	black smartcash/ tests/
	@echo "\n\033[1mSorting imports with isort...\033[0m"
	isort smartcash/ tests/

.PHONY: typecheck
typecheck:
	@echo "\n\033[1mRunning mypy...\033[0m"
	mypy smartcash/

.PHONY: check-all
check-all: lint format typecheck test
	@echo "\n\033[1;32mAll checks passed!\033[0m"

# Cleanup
.PHONY: clean
clean:
	@echo "\n\033[1mCleaning up...\033[0m"
	rm -rf .pytest_cache/ .coverage htmlcov/ .mypy_cache/ build/ dist/ *.egg-info/
