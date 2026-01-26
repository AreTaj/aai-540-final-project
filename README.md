# AAI-540 Final Project: Olist E-commerce ML System

Final Project for AAI 540 Machine Learning Ops. This project implements an end-to-end Machine Learning system using AWS SageMaker for predicting customer satisfaction on the Olist E-commerce dataset.

## Project Deliverables & Status

This repository is structured around the components required for the ML System Operation Validation.

### 1. Data Engineering & Feature Store
- [x] **Raw Data Collection**: Automated ingestion from Kaggle to S3 Datalake (`/raw`).
- [x] **Data Catalog**: Athena table setup for querying and validation.
- [x] **Exploratory Data Analysis (EDA)**: Analysis of order trends and data quality.
- [x] **Feature Engineering**: Creation of features (e.g., delivery time, review scores) and storage in SageMaker Feature Store.
- [x] **Data Splitting**: Stratified split into Train (40%), Validation (10%), Test (10%), and Production Holdout (40%).
- **Current Notebook**: `01_Data_Preparation.ipynb` (Budget-Optimized)

### 2. Model Training & Registry
- [ ] **Model Training**: TBD
- [ ] **Model Evaluation**: TBD
- [ ] **Model Registry**: TBD - Validating registration of model versions.

### 3. Deployment & Inference
- [ ] **Batch Inference / Endpoint**: TBD
- [ ] **Output Validation**: TBD

### 4. Pipelines & Automation (CI/CD)
- [ ] **SageMaker Pipelines**: TBD - Automated DAG definition.
- [ ] **CI/CD States**: TBD - Demonstration of successful and failed pipeline states.

### 5. Monitoring & Observability
- [ ] **Infrastructure Dashboards**: TBD
- [ ] **Model & Data Quality Reports**: TBD

## Repository Structure
* `01_Data_Preparation.ipynb`: Primary Data Preparation, Feature Engineering, and Feature Store ingestion pipeline. *Note: optimized for AWS Learner Labs (prevents redundant S3 uploads and Feature Group creates).*
