# 🛠️ Conversion of `.onnx` File to `.hef` File

> This guide provides a step-by-step process to convert an ONNX model to a HEF file using Hailo's Dataflow Compiler on a Linux system.  
> ⚙️ The entire process can be done on a laptop running Ubuntu 20.04 or 22.04 with Python 3.10.

---

## 📋 Prerequisites

Ensure the following dependencies are installed:

- Ubuntu 20.04 or 22.04
- Python 3.10
- Internet access for downloads
- A `.onnx` model file (e.g., `best.onnx`)
- A folder containing calibration images (e.g., `images/train`)
> ⚠️ **Important:** Make sure your calibration images are in the same resolution as your ONNX input shape (e.g., `640x640` or `1024x1024`). Mismatched sizes may cause poor calibration or inference issues.


---

## 🔧 Installation Steps

### 1. Update and Install System Packages

```bash
sudo apt update
sudo apt install -y python3-pip python3.10-venv
sudo apt install -y python3.10-dev python3.10-distutils python3-tk libfuse2 graphviz libgraphviz-dev
sudo apt install -y wslu
```

### 2. Install `pygraphviz`

```bash
sudo pip install pygraphviz
```

---

## 🧪 Environment Setup

### 3. Create and Activate a Virtual Environment

```bash
python3 -m venv hailodfc
source hailodfc/bin/activate
```

---

## 📦 Install Hailo Dataflow Compiler

### 4. Download the Compiler Wheel

- Go to [Hailo Developer Zone - Software Downloads](https://hailo.ai/developer-zone/software-downloads/)
- Select: **Linux** and **Python 3.10**
- Download the latest version of `hailo_dataflow_compiler-<version>-py3-none-linux_x86_64.whl`

> 💡 You can use `wslview .` to open the file explorer from WSL and move the downloaded file into your project folder.

### 5. Install the Wheel

```bash
pip install hailo_dataflow_compiler-<version>.whl
```

### 6. Verify Installation

```bash
hailo -h
pip freeze | grep hailo
```

---

## 🧬 Clone and Setup Hailo Model Zoo

### 7. Clone the Repository

```bash
git clone https://github.com/hailo-ai/hailo_model_zoo.git
cd hailo_model_zoo
git checkout tags/v2.12
pip install -e .
cd ..
```

---

## 📁 Prepare Files

Ensure the following files are in your working directory:

- `best.onnx` (your ONNX model)
- A folder of calibration images (e.g., `images/train`)

---

## 🚀 Compile the Model

### 8. Run the Compiler

```bash
hailomz compile yolov8n \
  --ckpt=best.onnx \
  --hw-arch hailo8 \
  --calib-path images/train \
  --classes 1 \
  --performance
```

> 📍 This will generate a `.hef` file which can be deployed to the Hailo-8 AI accelerator.

---

## ✅ Output

You should see output like:

```
[info] Saved HAR to: ./yolov8n.har
[info] HEF file written to: ./yolov8n.hef
```

---

## 🧯 Troubleshooting

- If `.whl` installation fails, ensure you selected the correct Python version and OS.
- If `pygraphviz` errors, try reinstalling system packages or building from source.
- If you see "Missing Graphviz", confirm both `graphviz` and `libgraphviz-dev` are installed.

---

## 📎 Notes

- Always activate your virtual environment before running `hailo` or compiling models.
- You can customize the `--classes` and `--calib-path` as per your dataset.

---

## 📄 License

This guide is for educational or personal use only. Refer to Hailo’s licensing terms for commercial deployment.









