# AAI-540 Final Project: Olist E-commerce ML System

Final Project for AAI 540 Machine Learning Ops. This project implements an end-to-end Machine Learning system using AWS SageMaker for predicting customer satisfaction on the Olist E-commerce dataset.

## Project Deliverables & Status

This repository is structured around the components required for the ML System Operation Validation.

### 1. Data Engineering & Feature Store
- [x] **Raw Data Collection**: Automated ingestion from Kaggle to S3 Datalake (`/raw`).
- [x] **Data Catalog**: Athena table setup for querying and validation.
- [x] **Exploratory Data Analysis (EDA)**: Analysis of order trends and data quality.
- [x] **Feature Engineering**: Creation of features (e.g., delivery time, review scores) and storage in SageMaker Feature Store.
- [x] **Data Splitting**: Time-based split (to prevent leakage) into Train (40%), Validation (10%), Test (10%), and Production Holdout (40%).
- **Current Notebook**: `01_Data_Preparation.ipynb` (Includes auto-dependency installation & budget optimizations)

### 2. Model Training & Registry
- [x] **Model Training**: XGBoost Baseline implemented (`02_Modeling.ipynb`).
- [x] **Model Evaluation**: Metrics (Accuracy, F1, ROC-AUC) calculation and comparison.
- [x] **Model Registry**: Training artifacts correctly stored in S3 with explicit registration logic.

### 3. Deployment & Inference
- [x] **Batch Inference / Endpoint**: Batch Transform job implemented for offline predictions on test set.
- [x] **Output Validation**: S3 output retrieval and verification of prediction results.

### 4. Pipelines & Automation (CI/CD)
- [x] **SageMaker Pipelines**: Automated DAG definition for end-to-end training and evaluation.
- [x] **CI/CD States**: Demonstration of successful and failed pipeline states using `ForceFailCI` parameters.

### 5. Monitoring & Observability
- [x] **Infrastructure Dashboards**: Centralized CloudWatch dashboard for endpoint health and processing jobs.
- [x] **Model & Data Quality Reports**: Automated violation detection for data drift and model performance decay.

## Repository Structure

### Primary Modular Pipeline
The core project logic is split into sequential, modular notebooks for easier maintainability and cost control:
* `01_Data_Preparation.ipynb`: Primary Data Preparation, Feature Engineering, and Feature Store ingestion pipeline. *Note: optimized for AWS Learner Labs.*
* `02_Modeling.ipynb`: Optimized Model Training and Batch Transform pipeline.
* `03_Deployment_and_Monitoring.ipynb`: Real-time endpoint deployment, Data Capture, and SageMaker Model Monitor setup.
* `03_Deployment_and_Monitoring_CLOUDWATCH.ipynb`: Extended monitoring with CloudWatch Alarms and Custom Dashboards.
* `04_CI_CD_Pipeline.ipynb`: Production-grade orchestration using SageMaker Pipelines, featuring parameterized execution and CI/CD validation.

### Parallel Development Paths
The following notebooks represent integrated, monolithic versions of the pipeline used during specific project modules or alternative execution paths:
* `AAI-540-G8-Project-Parallel-Path.ipynb`: Initial end-to-end integration testing.
* `AAI-540-G8-Project-Parallel-Path-Mod-5.ipynb`: Unified workflow focusing on Module 5 (Monitoring) deliverables.
* `AAI-540-G8-Project-Parallel-Path-Mod-6.ipynb`: Unified workflow focusing on Module 6 (CI/CD) deliverables.
* `AAI-540-G8-Project-Version2.ipynb` & `AAI-540-G8-Project.ipynb`: Earlier development iterations and team collaboration drafts.
