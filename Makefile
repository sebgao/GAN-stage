test:
	find . -name "*.pyc" | xargs rm
	find . -name "__pycache__" | xargs rm -r