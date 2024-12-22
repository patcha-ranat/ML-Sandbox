.SILENT: venv install

venv:
	python -m venv pyenv
	echo "Please, run 'source pyenv/Scripts/activate' before 'make install'"

install:
	pip install -r requirements.txt