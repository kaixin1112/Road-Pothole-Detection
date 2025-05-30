
# **Using DeGirum PySDK, DeGirum Tools, and Hailo Hardware**  

This md file provides a comprehensive guide on using **DeGirum PySDK**, **DeGirum Tools**, and **Hailo hardware** for efficient AI inference. These tools simplify edge AI development by enabling seamless integration, testing, and deployment of AI models on multiple hardware platforms, including **Hailo-8** and **Hailo-8L**.  

---

## **Table of Contents**  

1. [Introduction](#introduction)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation) 
4. [Additional Resources](#additional-resources) 

---

## **Introduction**  

DeGirum provides a powerful suite of tools to simplify the development and deployment of edge AI applications:  

- [**DeGirum PySDK**](https://github.com/DeGirum/PySDKExamples): The core library for integrating AI inference capabilities into applications.  
- [**DeGirum Tools**](https://github.com/DeGirum/degirum_tools): Utilities for benchmarking, streaming, and interacting with DeGirum's model zoo.  

These tools are designed to be hardware-agnostic, enabling developers to build scalable, flexible solutions without being locked into a specific platform.  

---

## **Prerequisites**  

- **Hailo Tools Installed**: Ensure that Hailo's tools and SDK are properly installed and configured. Refer to [Hailo's documentation](https://hailo.ai/) for detailed setup instructions. Also, enable the HailoRT Multi-Process service as per HailoRT documentation:  

  ```bash
  sudo systemctl enable --now hailort.service  # for Ubuntu
  ```  

- **Hailo Runtime Compatibility**:  
  DeGirum PySDK supports **Hailo Runtime versions 4.19.0, 4.20.0 and 4.21.0**. Ensure your Hailo environment is configured to use one of these versions.  

- **Python 3.9 or Later**: Ensure Python is installed on your system. You can check your Python version using:  

  ```bash
  python3 --version
  ```  

---

## **Installation**  

The best way to get started is to **clone this repository** and set up a virtual environment to keep dependencies organized. Follow these steps:  

### **1. Clone the Repository**  
```bash
git clone https://github.com/DeGirum/hailo_examples.git
cd hailo_examples
```  

### **2. Create a Virtual Environment**  
To keep the Python environment isolated, create a virtual environment:  

#### **Linux/macOS**  
```bash
python3 -m venv degirum_env
source degirum_env/bin/activate
```  

#### **Windows**  
```bash
python3 -m venv degirum_env
degirum_env\Scripts\activate
```  

### **3. Install Required Dependencies**  
Install all necessary packages from `requirements.txt`:  

```bash
pip install -r requirements.txt
```  

---

### **4. Add Virtual Environment to Jupyter**  

If you plan to use **Jupyter Notebooks**, ensure the virtual environment is available as a Jupyter kernel.  

#### **Step 1: Activate the Virtual Environment (if not already active)**  
If you are not already inside the virtual environment, activate it:  

**Linux/macOS:**  
```bash
source degirum_env/bin/activate
```  

**Windows:**  
```bash
degirum_env\Scripts\activate
```  

#### **Step 2: Ensure the Virtual Environment is Available in Jupyter**  
Since `notebook` and `ipykernel` are already installed via `requirements.txt`, simply run:  

```bash
python -m ipykernel install --user --name=degirum_env --display-name "Python (degirum_env)"
```  

This ensures that Jupyter recognizes the virtual environment as an available kernel.  

---

### **5. Verify Installation**  

To ensure that everything is set up correctly, run the provided test script:  

```bash
python test.py
```  

This script will:  
- Check system information.  
- Verify that Hailo hardware is recognized.  
- Load and run inference with a sample AI model.  

If the test runs successfully, your environment is properly configured.  

## Additional Resources

- [Hailo Model Zoo](./hailo_model_zoo.md): Explore the full list of models optimized for Hailo hardware.
- [DeGirum Documentation](https://docs.degirum.com)
- [Hailo Documentation](https://hailo.ai/)
