.PHONY: help install test reproduce_all reproduce_classification clean

help:
	@echo "Adaptive Alpha-GoLU Command Suite"
	@echo "----------------------------------"
	@echo "make install                 - Install requirements"
	@echo "make test                    - Run quick unit & baseline tests"
	@echo "make reproduce_classification - Run classification benchmarks across seeds"
	@echo "make reproduce_all           - Run full suite (All 6 tasks across seeds)"

install:
	pip install -r requirements.txt

test:
	python -m unittest discover tests/

reproduce_classification:
	python cli.py run --task classification --activation alpha_golu --seeds 42 123 999

reproduce_all:
	python cli.py run_all --seeds 42 123 999

clean:
	rm -rf __pycache__ outputs/*.png .pytest_cache
