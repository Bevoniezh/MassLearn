# MassLearn Environment Setup

This project uses a Dash-based interface together with scientific Python tooling for mass spectrometry analysis. The instructions below help you create a reproducible environment that works on both Windows and Linux machines.

## 1. Prerequisites

- **Python 3.10** (or a Conda distribution such as [Miniconda](https://docs.conda.io/en/latest/miniconda.html)).
- Git (if you want to clone the repository directly).
- On Linux desktops you may need the system Qt dependencies for PyQt to display correctly (`sudo apt install libqt5gui5 libqt5webengine5` on Debian/Ubuntu-based distributions).
- External vendor tools used by MassLearn workflows:
  - **MZmine** from the [MZmine project](https://mzmine.github.io/).

See the section "External tools: download, licensing, and first-run setup" below for detailed installation guidance and licensing information.

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
   python "MassLearn 2.3.py"
   ```

### Using Micromamba on Windows

If you prefer Micromamba, you can reuse the same `environment.yml` without modification. The steps below assume Windows PowerShell:

1. **Install Micromamba**
   - Download the latest Windows release from the [Micromamba GitHub releases](https://github.com/mamba-org/micromamba-releases).
   - Extract the archive and add the folder containing `micromamba.exe` to your `PATH`, or place the executable somewhere convenient such as `C:\micromamba`.
   - Optionally initialize shell hooks so `micromamba` commands are available automatically:
     ```powershell
     .\micromamba.exe shell init -s powershell -p "$Env:USERPROFILE\micromamba"
     ```
     Reload the PowerShell session afterwards.
2. **Create the environment** (from the repository root):
   ```powershell
   micromamba create -f environment.yml
   ```
   Micromamba will honor the named environment (`masslearn`) from the YAML file and create it under `%USERPROFILE%\micromamba\envs` unless you configured a custom root.
3. **Activate the environment** each time you work on the project:
   ```powershell
   micromamba activate masslearn
   ```
4. **Run the application** once the prompt shows `(masslearn)`:
   ```powershell
   python "MassLearn 2.3.py"
   ```
5. **Update the environment** in the future when dependencies change:
   ```powershell
   micromamba update -f environment.yml
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
   python "MassLearn 2.3.py"
   ```

## 4. Environment notes

- The dependency manifests include every third-party package imported by the Python sources, including `ezodf` for `.ods` batch templates, `psutil` for resource monitoring, and `lxml` for XML manipulation.
- PyQt is required for features that open Plotly content in a desktop window. If you only use the web UI you can omit `PyQt5`/`PyQtWebEngine`.
- `tkinter` is part of the standard Python distribution on Windows and macOS. On some Linux distributions you may need to install it separately (e.g. `sudo apt install python3-tk`).
- When running on a headless server you might disable or skip the modules that rely on GUI components.

## 5. Updating the environment

Whenever you pull new changes that modify dependencies, re-run either `conda env update -f environment.yml` or `pip install -r requirements.txt` to keep your environment synchronized.

## 6. External tools: download, licensing, and first-run setup

MassLearn integrates with three Windows desktop applications. Install them before launching the login page so that you can register their paths when prompted.

### 6.1 MZmine 3

- Download MZmine from the [official release page](https://mzmine.github.io/download.html) and follow the installer instructions.
- MZmine is released under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). Make sure you can comply with the requirements of GPLv3 (or obtain a professional license if needed).
- After installation, **launch MZmine once manually** so the license activation dialog can complete (choose the academic or professional license type as appropriate). MassLearn will not be able to start MZmine automatically until this first-run activation succeeds.

### 6.2 Registering executable paths in MassLearn

1. Start MassLearn and open the **Login** page.
2. For first-time configuration—or whenever you update or reinstall any of the tools—click each software icon (SeeMS, MZmine, and MSConvert) displayed under the "Powered by" section.
3. Enter or paste the full path to the corresponding executable (`MZmine.exe`) and click **Confirm file**.
4. MassLearn stores the paths in `Cache/software_path_dash.dat`. If an executable is moved or replaced, repeat the steps above to update the stored path.

If MassLearn cannot launch a tool (for example after an update), return to the Login page and re-register the executable path.
