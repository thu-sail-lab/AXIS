# AXIS: Explainable Time Series Anomaly Detection with Large Language Models

## Overview

AXIS is an explainable time series anomaly detection framework that leverages Large Language Models to provide natural language explanations for detected anomalies.

## Project Structure

```
AXIS/
├── src/models/AXIS/
│   ├── AXIS.py                 # Main model implementation
│   ├── AXIS_test.py           # Testing framework
│   ├── dataset.py             # Dataset utilities
│   ├── Pretrain_ts_encoder.py # Time series encoder
│   └── ts_encoder_bi_bias.py  # Encoder components
├── experiments/
│   ├── configs/               # Configuration files
│   ├── checkpoints/           # Model checkpoints
│   └── logs/                  # Training and testing logs
├── data/
│   └── AXIS_qa_test/          # Test dataset
├── requirements.txt           # Python dependencies
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd AXIS
```

### 2. Create conda environment and install dependencies

```bash
# Create conda environment
conda create -n AXIS python=3.11

# Activate environment
conda activate AXIS

# Install dependencies
pip install -r requirements.txt
```



### 3. Download and extract model checkpoints

Download the pre-trained model checkpoints from Hugging Face:

```bash
huggingface-cli download thu-sail-lab/TimeSemantic checkpoints.zip --local-dir ./experiments
```

Extract the downloaded checkpoint file:

```bash
cd experiments
unzip checkpoints.zip
cd ..
```

## Usage

### Run Testing and Generate Results

1. **Set environment variables**

```bash
export HF_TOKEN="your_huggingface_token"
export CUDA_VISIBLE_DEVICES=0
```

2. **Run test script**

```bash
# Set PYTHONPATH and run test
python -m src.models.AXIS.AXIS_test
```

3. **View results**

Test results are saved in:

- **Log files**: `experiments/logs/AXIS/axis_test_YYYYMMDD_HHMMSS.txt`
- **Detailed results**: `experiments/logs/AXIS/<model_name>/results_YYYYMMDD_HHMMSS/`
  - Individual question results in YAML format (`question_XXXXXX.yaml`)
  - Test summary in `test_summary.yaml`
