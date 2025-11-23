# MLOps Kubeflow Assignment – Boston Housing Pipeline

## 1. Project Overview

This project implements an end-to-end **MLOps workflow** for a **regression** task using the **Boston Housing dataset**.  
The goal is to predict house prices based on multiple features (e.g., rooms, crime rate, tax, etc.) while applying proper MLOps practices:

- **DVC** for data versioning and remote storage  
- **Kubeflow Pipelines (KFP)** on **Minikube** for orchestrating the ML workflow  
- **Jenkins CI** (or GitHub Actions) to automatically compile and validate the pipeline on every change  

The core ML workflow consists of four main steps:

1. **Data Extraction** – Download versioned data using DVC  
2. **Data Preprocessing** – Split into train/test and scale features  
3. **Model Training** – Train a RandomForest regressor  
4. **Model Evaluation** – Evaluate performance (e.g., RMSE, R²) and save metrics  

---

## 2. Setup Instructions

### 2.1. Prerequisites

- **OS:** Windows 11  
- **Python:** 3.10  
- **Tools installed:**
  - [Git](https://git-scm.com/)
  - [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - [Minikube](https://minikube.sigs.k8s.io/docs/start/)
  - [kubectl](https://kubernetes.io/docs/tasks/tools/)
  - [DVC](https://dvc.org/doc/install)
  - [Jenkins](https://www.jenkins.io/) (for Task 4)

Clone the repository:

```bash
git clone https://github.com/<YOUR_USERNAME>/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment
