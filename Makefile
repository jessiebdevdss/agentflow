.PHONY: install test demo eval
install:
	python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[all]"
test:
	. .venv/bin/activate && pytest -q
demo:
	. .venv/bin/activate && agentflow run "Compare LangGraph and CrewAI for production multi-agent systems" --mock
eval:
	. .venv/bin/activate && agentflow eval --mock
