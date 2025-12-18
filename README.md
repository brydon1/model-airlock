# Model-Airlock 🚀

A Python CLI tool designed to standardize the handoff between local research experimentation and cloud storage. 

## 🎯 The Motivation
In machine learning and robotics workflows, moving from "It works in my Jupyter Notebook" to "It's ready for the pipeline" is often a friction point. `Model-Airlock` acts as a developer tool that ensures artifacts meet structural and mathematical prerequisites before they are allowed into the S3 storage bucket.

**Role:** Architecture & Tooling Demo  
**Tech Stack:** Python 3.11, Typer (CLI), Boto3 (AWS), Pydantic (Validation), Pytest.

## 🛠 Key Features

* **Schema Validation:** Enforces strict typing on model metadata configuration files using Pydantic, preventing "magic number" configuration errors downstream.
* **Sanity Checks:** Performs a mock mathematical validation (e.g., ensuring tensor output dimensions match expected configuration) before network calls are made.
* **Cloud Persistence:** Handles secure upload of artifacts to AWS S3 with automatic semantic versioning tagging.
* **Developer Experience:** Provides rich terminal output and logging to help researchers debug validation failures quickly.

## Prerequisites

Python 3.9+ (Tested on Python 3.13)

AWS CLI (Optional for deployment)

## 💻 Usage

**1. Installation**

Bash
```bash
git clone https://github.com/brydon1/model-airlock.git
cd model-airlock
pip install -r requirements.txt
```

**2. Create a Virtual Environment (Recommended)**

Create a virtual environment named venv and activate it.

Bash
```bash
python -m venv venv

# Windows (Git Bash / Command Prompt):
source venv/Scripts/activate

# macOS / Linux:
source venv/bin/activate
```

**3. Install Dependencies**

If using a virtual machine, these will live in an isolated folder instead of globally on your machine.

Bash
```bash
pip install -r requirements.txt
```

**4. Run a Validation & Upload**

Bash
```bash
# Validates the model_config.json and uploads to the 'robotics-staging' bucket
python main.py deploy \
  --model-file examples/model.pt \
  --config examples/model_config.json \
  --bucket robotics-staging \
  --dry-run
```

## 🏗 Architecture Decisions

### Why `Typer`?

I chose Typer over Argparse to utilize Python type hints for a cleaner, self-documenting codebase that is easier for other developers to maintain.

### Why Pre-Upload Validation?

Checking constraints locally saves cloud bandwidth and prevents the "Garbage In, Garbage Out" problem in the training pipeline.

## 🧪 Testing

Includes a suite of unit tests for the validation logic.

Bash
```bash
pytest tests/
```

### Generating Test Data
To verify the validation logic, you can generate valid dummy models for PyTorch, ONNX, and Sklearn.

Bash
```bash
# Install generation dependencies (optional)
pip install -r requirements-dev.txt
# Generate dummy model files
python generate_dummies.py
```

## 👤 About Me

I am a Software Engineer with a background in Mathematics and 5 years of experience building stable middleware and internal tooling. I built this project to demonstrate how I approach building guardrails for complex development ecosystems.