.PHONY: style

style:
	black .
	isort . --profile=black