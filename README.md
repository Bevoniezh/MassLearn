# MassLearn Environment Setup

This project uses a Dash-based interface together with scientific Python tooling for mass spectrometry analysis. The instructions below help you create a reproducible environment that works on both Windows and Linux machines.

## 1. Prerequisites

- **Python 3.10** (or a Conda distribution such as [Miniconda](https://docs.conda.io/en/latest/miniconda.html)).
- Git (if you want to clone the repository directly).
- On Linux desktops you may need the system Qt dependencies for PyQt to display correctly (`sudo apt install libqt5gui5 libqt5webengine5` on Debian/Ubuntu-based distributions).

## 2. Option A – Conda environment (recommended for cross-platform)

1. Install Miniconda/Anaconda if it is not already available.
2. Open a terminal (or Anaconda Prompt on Windows) and navigate to the project folder.
3. Create the environment:
   ```bash
   conda env create -f environment.yml
   ```
4. Activate it:
   ```bash
   conda activate masslearn
   ```
5. Launch the application:
   ```bash
   python "MassLearn 2.2.py"
   ```

## 3. Option B – Python virtual environment with pip

1. Ensure Python 3.10 is installed and available in your PATH.
2. Create and activate a virtual environment:
   ```bash
   # Linux / macOS
   python -m venv .venv
   source .venv/bin/activate

   # Windows
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Upgrade pip and install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python "MassLearn 2.2.py"
   ```

## 4. Environment notes

- The dependency manifests include every third-party package imported by the Python sources, including `ezodf` for `.ods` batch templates, `psutil` for resource monitoring, and `lxml` for XML manipulation.
- PyQt is required for features that open Plotly content in a desktop window. If you only use the web UI you can omit `PyQt5`/`PyQtWebEngine`.
- `tkinter` is part of the standard Python distribution on Windows and macOS. On some Linux distributions you may need to install it separately (e.g. `sudo apt install python3-tk`).
- When running on a headless server you might disable or skip the modules that rely on GUI components.

## 5. Updating the environment

Whenever you pull new changes that modify dependencies, re-run either `conda env update -f environment.yml` or `pip install -r requirements.txt` to keep your environment synchronized.
