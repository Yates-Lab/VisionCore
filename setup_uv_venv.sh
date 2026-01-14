#!/bin/bash
# This script installs an IPython kernel for the virtual environment located in the .venv directory.
# cd into the project directory
uv sync
uv pip install -e ../DataYatesV1
uv pip install -e ../DataRowleyV1V2
cd "$(dirname "$0")"
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=vision-core --display-name "VisionCore"

