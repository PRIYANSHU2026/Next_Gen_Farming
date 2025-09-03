# Soil Quality and Fertility Prediction System Makefile

.PHONY: setup run test clean lint

# Setup the project
setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

# Run the application
run:
	. venv/bin/activate && streamlit run src/soil_health_streamlit.py

# Run tests
test:
	. venv/bin/activate && python -m pytest tests/

# Clean up generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Lint the code
lint:
	. venv/bin/activate && flake8 src/ tests/
	. venv/bin/activate && black src/ tests/