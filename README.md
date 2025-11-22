# Diabetes Prediction

A machine learning project to predict diabetes using clinical data, featuring a robust data pipeline, model training, and both API and interactive interfaces.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
	- [1. Jupyter Notebook](#1-jupyter-notebook)
	- [2. Model Training Script](#2-model-training-script)
	- [3. FastAPI Web API](#3-fastapi-web-api)
- [API Endpoints](#api-endpoints)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Author](#author)
- [License](#license)

---

## Project Overview

This project predicts the likelihood of diabetes in patients based on medical attributes. It includes:

- Data cleaning and preprocessing
- Exploratory data analysis
- Model selection and hyperparameter tuning
- Deployment as a FastAPI web service and Gradio interface

## Dataset

- **Source:** [Kaggle Diabetes Dataset](https://www.kaggle.com/johndasilva/diabetes)
- **Location:** `Dataset/kaggle_diabetes.csv`
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome

## Features

- Data cleaning (handling missing/zero values)
- Feature scaling
- Model selection (Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Gradient Boosting)
- Hyperparameter tuning with GridSearchCV
- Best model: Random Forest (F1 Score ≈ 0.94)
- REST API for predictions (FastAPI)
- Interactive UI (Gradio)

## Installation

1. **Clone the repository:**
	 ```sh
	 git clone <repo-url>
	 cd Diabetes-Prediction
	 ```

2. **Create and activate a virtual environment (optional but recommended):**
	 ```sh
	 python -m venv env
	 .\env\Scripts\activate
	 ```

3. **Install dependencies:**
	 ```sh
	 pip install -r requirements.txt
	 ```

## Usage

### 1. Jupyter Notebook

Explore and run the full data science workflow in `Notebooks/Diabetes_Prediction.ipynb`.

### 2. Model Training Script

To retrain and save the model:
```sh
python model/Diabetes_Classification.py
```
This will output `diabetes_model.pkl` in the `model/` directory.

### 3. FastAPI Web API

To launch the API server:
```sh
python run.py
```
The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

- `GET /Home`  
	Health check endpoint.

- `POST /Diabetes_Classification`  
	Predict diabetes from input features.  
	**Request Body Example:**
	```json
	{
		"Pregnancies": 2,
		"Glucose": 120,
		"BloodPressure": 70,
		"SkinThickness": 25,
		"Insulin": 90,
		"BMI": 30.5,
		"DiabetesPedigreeFunction": 0.52,
		"Age": 33
	}
	```
	**Response:**
	```json
	{ "Message": "You Have Diabetes" }
	```

## Requirements

See `requirements.txt` for all dependencies. Key packages:
- fastapi
- scikit-learn
- pandas
- numpy
- joblib
- gradio

## Project Structure

```
.
├── Dataset/
│   └── kaggle_diabetes.csv
├── model/
│   ├── Diabetes_Classification.py
│   ├── app.py
│   └── diabetes_model.pkl
├── Notebooks/
│   └── Diabetes_Prediction.ipynb
├── run.py
├── requirements.txt
└── README.md
```

## Author

- Mohamed Soliman


