.PHONY: install run test check demo
install:
	python -m pip install -e '.[dev]'
run:
	research-assist serve
demo:
	research-assist seed
	research-assist ask "RAG và Self-RAG khác nhau như thế nào?"
test:
	pytest --cov=research_assist --cov-report=term-missing
check:
	ruff check src tests scripts
	ruff format --check src tests scripts
	mypy src/research_assist
