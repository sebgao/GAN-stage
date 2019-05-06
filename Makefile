clean:
	find . -name "*.pyc" | xargs rm
	find . -name "__pycache__" | xargs rm -r

test:
	python main.py

git:
	make clean
	git add .
	git commit -m '.'