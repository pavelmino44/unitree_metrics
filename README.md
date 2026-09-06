# unitree_metrics

Tools for visualizing and analyzing reinforcement learning training metrics from TensorBoard logs.

The repository contains Python scripts and Jupyter notebooks for processing training results, comparing multiple runs, smoothing metrics, filtering outliers, and generating training curves.

## Features

* Load scalar metrics from TensorBoard event files.
* Combine metrics from multiple training runs.
* Visualize mean reward and episode length.
* Apply exponential moving average (EMA) smoothing.
* Filter abnormal spikes in training curves.
* Remove selected regions from plots for visualization.
* Analyze and compare reinforcement learning experiments.

## Requirements

* Python 3
* NumPy
* Matplotlib
* TensorBoard
* Jupyter Notebook or JupyterLab (optional, for running the notebooks)

## Installation

Clone the repository:

```bash
git clone git@github.com:pavelmino44/unitree_metrics.git
cd unitree_metrics
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Visualizing training metrics

The main script reads TensorBoard logs from several training runs and generates plots for:

* mean reward;
* mean episode length.

By default, the script expects the following directory structure:

```text
unitree_metrics/
├── main.py
└── logs/
    ├── run_1/
    ├── run_2/
    ├── run_3/
    ├── run_4/
    └── run_5/
```

Each run directory should contain TensorBoard event files.

Run the visualizer with:

```bash
python3 main.py
```

The script will load the metrics, merge the selected runs, apply data processing and smoothing, and display the resulting plots.

## Jupyter Notebooks

The repository also contains two notebooks:

### `metrics_visualizer.ipynb`

Interactive visualization of training metrics loaded from TensorBoard event files.

### `data_analysis.ipynb`

Additional analysis of reinforcement learning training data.

The notebooks can be opened with Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

## Training Logs

Training logs and model checkpoints are intentionally **not included in the repository**.

Large experiment artifacts such as:

* TensorBoard event files;
* PyTorch checkpoints (`.pt`);
* ONNX models;
* experiment configuration files;
* exported policies

should be stored separately.

The `logs/` directory is included in `.gitignore` so local experiment results can be used without accidentally committing them to Git.

For example, after cloning the repository, local experiment results can be placed in:

```text
unitree_metrics/
└── logs/
    ├── run_1/
    ├── run_2/
    └── ...
```

## Project Structure

```text
unitree_metrics/
├── main.py
├── data_analysis.ipynb
├── metrics_visualizer.ipynb
├── pics/
│   ├── Figure_1.png
│   └── Figure_2.png
├── requirements.txt
├── .gitignore
└── logs/                 # local, not tracked by Git
```
