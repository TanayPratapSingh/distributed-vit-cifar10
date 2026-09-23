VENV := .venv/bin
.PHONY: install test smoke laptop gloo kaggle report dashboard verify gate clean

install:
	python3 -m venv .venv
	$(VENV)/pip install -q --upgrade pip
	$(VENV)/pip install -q -e ".[dev]"

test:
	$(VENV)/python -m pytest

smoke:
	$(VENV)/python -m dvit.sweep smoke --continue-on-error
	$(MAKE) report

laptop:
	$(VENV)/python -m dvit.sweep laptop --continue-on-error
	$(MAKE) report

gloo:
	$(VENV)/python -m dvit.sweep gloo-overhead --continue-on-error
	$(MAKE) report

kaggle:
	$(VENV)/python -m dvit.sweep kaggle --continue-on-error
	$(MAKE) report

report dashboard:
	$(VENV)/python -m dvit.report

verify:
	$(VENV)/python -m dvit.verify_claims

gate: test verify
	@echo "tests pass and every unit bearing number traces to an artifact"

clean:
	rm -rf runs/*.json dashboard/index.html .pytest_cache
