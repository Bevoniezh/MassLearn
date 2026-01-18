# MassLearn Environment Setup

This project uses a Dash-based interface together with scientific Python tooling for mass spectrometry analysis. The instructions below help you create a reproducible environment that works on both Windows and Linux machines.

## 1. Prerequisites (step-by-step, beginner friendly)

These steps assume you are installing everything from scratch. If you already have a tool installed, you can skip that tool's steps.

### 1.1 Install Git (required to clone the project)

**Windows**
1. Open your web browser and go to the [Git for Windows download page](https://git-scm.com/download/win).
2. Download and run the installer.
3. During setup, keep the defaults unless you know you need something different.
4. After the install finishes, open **Start Menu → Git Bash** (this is the terminal you will use for Git commands).

**macOS**
1. Open **Applications → Utilities → Terminal**.
2. Run:
   ```bash
   git --version
   ```
3. If Git is not installed, macOS will prompt you to install the **Command Line Tools**. Click **Install** and wait for it to finish.

**Linux (Ubuntu/Debian)**
1. Open your **Terminal** application.
2. Run:
   ```bash
   sudo apt update
   sudo apt install git
   ```
3. Confirm Git works:
   ```bash
   git --version
   ```

### 1.2 Install Miniconda (recommended Python distribution)

Miniconda gives you Python 3.10 and the `conda` command used in the setup steps below.

**Windows**
1. Open a browser and download the **Windows 64-bit** Miniconda installer from the [official Miniconda page](https://docs.conda.io/en/latest/miniconda.html).
2. Run the installer.
3. When asked, choose **Just Me** and keep the default install location.
4. On the "Advanced Installation Options" screen, check **Add Miniconda to my PATH** if you are comfortable with it; otherwise leave it unchecked and use **Anaconda Prompt** (recommended for beginners).
5. Open **Start Menu → Anaconda Prompt** (this is the terminal you will use for conda commands).

**macOS**
1. Download the **macOS** Miniconda installer from the [official Miniconda page](https://docs.conda.io/en/latest/miniconda.html).
2. Open **Terminal** and run the downloaded `.sh` installer:
   ```bash
   bash ~/Downloads/Miniconda3-latest-MacOSX-x86_64.sh
   ```
   (Adjust the filename if you downloaded the Apple Silicon installer.)
3. Follow the prompts and accept the default install location.
4. Close and reopen Terminal, then confirm:
   ```bash
   conda --version
   ```

**Linux**
1. Download the **Linux 64-bit** Miniconda installer from the [official Miniconda page](https://docs.conda.io/en/latest/miniconda.html).
2. Open Terminal and run:
   ```bash
   bash ~/Downloads/Miniconda3-latest-Linux-x86_64.sh
   ```
3. Follow the prompts and accept the default install location.
4. Close and reopen Terminal, then confirm:
   ```bash
   conda --version
   ```

### 1.3 (Linux only) Install Qt system libraries for PyQt

If you are on a Linux desktop and plan to use the PyQt features, install the Qt system libraries:
```bash
sudo apt install libqt5gui5 libqt5webengine5
```

### 1.4 External vendor tool (MZmine)

MassLearn workflows use **MZmine**. Install it before running MassLearn so the app can locate it later.

1. Download MZmine from the [MZmine project site](https://mzmine.github.io/).
2. Install it using the installer for your OS.
3. Launch MZmine once to finish any first-run setup or license activation.

See the section "External tools: download, licensing, and first-run setup" below for detailed licensing and first-run notes.

## 2. Get the project from GitHub (step-by-step)

These steps explain how to clone the project using Git.

1. Decide where you want the project folder to live (e.g. `Documents` on Windows or `~/Projects` on macOS/Linux).
2. Open your terminal:
   - **Windows**: open **Git Bash**.
   - **macOS**: open **Terminal**.
   - **Linux**: open **Terminal**.
3. Move into the folder where you want to keep the project, for example:
   ```bash
   cd ~/Documents
   ```
4. Clone the repository (replace the URL with the real one if needed):
   ```bash
   git clone <REPO_URL_HERE>
   ```
5. Enter the project folder:
   ```bash
   cd MassLearn
   ```

At this point your terminal should be inside the MassLearn project folder. Keep this terminal open for the next steps.

## 3. Option A – Conda environment (recommended for cross-platform)

These steps assume you are **already in the MassLearn project folder**.

1. Open the correct terminal:
   - **Windows**: **Anaconda Prompt** (or the same terminal where `conda --version` works).
   - **macOS/Linux**: your normal **Terminal**.
2. Create the environment:
   ```bash
   conda env create -f environment.yml
   ```
3. Activate it (you should see `(masslearn)` appear in your prompt):
   ```bash
   conda activate masslearn
   ```
4. Launch the application:
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

## 4. Option B – Python virtual environment with pip

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

## 5. Environment notes

- The dependency manifests include every third-party package imported by the Python sources, including `ezodf` for `.ods` batch templates, `psutil` for resource monitoring, and `lxml` for XML manipulation.
- PyQt is required for features that open Plotly content in a desktop window. If you only use the web UI you can omit `PyQt5`/`PyQtWebEngine`.
- `tkinter` is part of the standard Python distribution on Windows and macOS. On some Linux distributions you may need to install it separately (e.g. `sudo apt install python3-tk`).
- When running on a headless server you might disable or skip the modules that rely on GUI components.

## 6. Updating the environment

Whenever you pull new changes that modify dependencies, re-run either `conda env update -f environment.yml` or `pip install -r requirements.txt` to keep your environment synchronized.

## 7. External tools: download, licensing, and first-run setup

MassLearn integrates with three Windows desktop applications. Install them before launching the login page so that you can register their paths when prompted.

### 7.1 MZmine 3

- Download MZmine from the [official release page](https://mzmine.github.io/download.html) and follow the installer instructions.
- MZmine is released under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). Make sure you can comply with the requirements of GPLv3 (or obtain a professional license if needed).
- After installation, **launch MZmine once manually** so the license activation dialog can complete (choose the academic or professional license type as appropriate). MassLearn will not be able to start MZmine automatically until this first-run activation succeeds.

### 7.2 Registering executable paths in MassLearn

1. Start MassLearn and open the **Login** page.
2. For first-time configuration—or whenever you update or reinstall any of the tools—click each software icon (SeeMS, MZmine, and MSConvert) displayed under the "Powered by" section.
3. Enter or paste the full path to the corresponding executable (`MZmine.exe`) and click **Confirm file**.
4. MassLearn stores the paths in `Cache/software_path_dash.dat`. If an executable is moved or replaced, repeat the steps above to update the stored path.

If MassLearn cannot launch a tool (for example after an update), return to the Login page and re-register the executable path.
