.PHONY: pdf stats check-stats empirical-stats figures audit clean
.DEFAULT_GOAL := pdf
SCI_PYTHON ?= uv run --project ../.. --no-sync python

stats:
	$(SCI_PYTHON) sync_stats.py

check-stats:
	$(SCI_PYTHON) sync_stats.py --check

empirical-stats:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/visioncore-mpl $(SCI_PYTHON) export_empirical_stats.py
	$(SCI_PYTHON) sync_stats.py

figures:
	$(SCI_PYTHON) render_figures.py

audit: check-stats pdf
	$(SCI_PYTHON) audit_figures.py

pdf:
	latexmk -pdf -pdflatex='pdflatex -no-shell-escape %O %S' -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1 -outdir=build main.tex

clean:
	latexmk -C -outdir=build main.tex
