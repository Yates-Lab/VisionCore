# VisionCore
multidataset training for digital twin models of visual cortex

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12.

```bash
# Clone the repository
git clone https://github.com/Yates-Lab/VisionCore.git
cd VisionCore

# Install dependencies
uv sync
```

### Optional: Data packages

To include the data loading packages, first clone the data repositories as siblings to VisionCore:

```bash
cd ..
git clone https://github.com/Yates-Lab/DataYatesV1.git
git clone https://github.com/Yates-Lab/DataRowleyV1V2.git
cd VisionCore
```

Then install with the `data` extra:

```bash
uv sync --extra data
```
