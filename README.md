# Smart Grocery Spoilage Prediction using Model Routing and Counterfactual Explanations

An end-to-end **MLOps and Machine Learning project** that predicts whether a grocery item is likely to spoil quickly during delivery.

The system uses a **model-routing architecture** where a router model selects one of three specialist models based on product fragility. It also provides a simple **counterfactual explanation** showing what could be changed to reduce the predicted spoilage risk.

> **Project level:** College / Academic MLOps Project
> **Focus:** Machine Learning, Model Routing, Explainable AI, MLOps

---

## Overview

Grocery products have different spoilage characteristics. For example, milk and berries are much more sensitive to delivery conditions than potatoes or onions.

Instead of using one model for every product, this project divides products into three categories:

* **Hardy** — potato, onion, apple
* **Normal** — tomato, carrot, banana
* **Fragile** — milk, berries, leafy vegetables

A **Random Forest router** first determines the appropriate category. The corresponding specialist model then predicts the spoilage risk.

The API also generates a simple counterfactual explanation such as:

> If temperature is reduced, the predicted spoilage risk decreases to Low.

---

## Key Features

* Model routing using a Random Forest classifier
* Three specialist models for different product categories
* Spoilage risk prediction: **Low / High**
* Manual counterfactual explanations
* Synthetic dataset generation
* MLflow experiment tracking
* Prefect training pipeline
* Evidently-based data drift monitoring
* FastAPI prediction API
* Pydantic request validation
* Automated testing with Pytest
* GitHub Actions CI pipeline

---

## System Architecture

```text
                    User Input
                        |
                        v
                FastAPI /predict
                        |
                        v
              +-------------------+
              |   Router Model    |
              |   Random Forest   |
              +-------------------+
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
       Hardy         Normal        Fragile
     Specialist    Specialist    Specialist
     Logistic RF    Random Forest    XGBoost
          |             |             |
          +-------------+-------------+
                        |
                        v
              Spoilage Risk
                 Low / High
                        |
                        v
          Counterfactual Explanation
                        |
                        v
                  JSON Response
```

---

## Machine Learning Approach

### 1. Router Model

The router is a **Random Forest classifier**.

It predicts one of:

```text
hardy
normal
fragile
```

The router uses:

* Temperature
* Humidity
* Delivery distance
* Initial shelf life

The product name itself is not directly provided to the router.

---

### 2. Specialist Models

After routing, one of three specialist models predicts the final spoilage risk.

| Specialist | Algorithm           | Prediction    |
| ---------- | ------------------- | ------------- |
| Hardy      | Logistic Regression | Spoilage risk |
| Normal     | Random Forest       | Spoilage risk |
| Fragile    | XGBoost             | Spoilage risk |

Each specialist uses the following numerical features:

* `temperature`
* `humidity`
* `distance_km`
* `initial_shelf_life`
* `packaging_quality`

---

## Dataset

The project uses a **synthetically generated dataset** because a suitable public grocery-spoilage dataset was not available for this project.

The dataset is generated using `src/data_generator.py`.

The generation rules include:

* Higher temperature → faster spoilage
* Higher humidity → faster spoilage
* Longer delivery distance → faster spoilage
* Better packaging → slower spoilage
* Lower shelf life → higher spoilage risk

Random noise is added to make the dataset less perfectly deterministic.

Approximately **4,000 records** are generated.

### Dataset Features

| Feature              | Description                   |
| -------------------- | ----------------------------- |
| `temperature`        | Delivery temperature in °C    |
| `humidity`           | Humidity percentage           |
| `distance_km`        | Delivery distance             |
| `product_type`       | Grocery product type          |
| `category`           | Hardy / Normal / Fragile      |
| `initial_shelf_life` | Initial shelf life in days    |
| `packaging_quality`  | Packaging quality from 1 to 3 |
| `spoilage_days`      | Estimated days until spoilage |
| `spoilage_risk`      | Target: 0 = Low, 1 = High     |

The spoilage risk is determined relative to the item's shelf life. An item is considered high risk when its estimated spoilage occurs within approximately **30% of its shelf life**.

---

## Counterfactual Explanations

The project includes a simple custom counterfactual explanation system implemented in:

```text
src/counterfactual.py
```

The function:

```python
find_counterfactual(input_data, model)
```

works as follows:

1. Get the current model prediction.
2. If the risk is already Low, no explanation is required.
3. Try changing temperature, humidity, or distance.
4. Re-run the model after each change.
5. Return the first change that converts the prediction from High to Low.

Example:

```text
If temperature is reduced from 30°C to 15°C,
the predicted spoilage risk decreases to Low.
```

This project intentionally does not use SHAP, LIME, or DiCE. The goal is to provide a simple and actionable **"what-if" explanation**.

---

## MLOps Components

### MLflow

MLflow is used for experiment tracking.

The training process records:

* Model name
* Accuracy
* Precision
* Recall
* F1 score

Run:

```bash
mlflow ui
```

to view the experiment dashboard.

MLflow data is stored locally in the `mlruns/` directory.

---

### Prefect

Prefect is used to organize the training process into a pipeline:

```text
Generate Data
      ↓
Train Models
      ↓
Save Models
```

The data-generation task also demonstrates Prefect's retry functionality.

Run:

```bash
python src/pipeline.py
```

---

### Evidently

Evidently is used for a basic data-drift check.

The monitoring script compares:

```text
Reference Data
      ↓
Training Dataset

Current Data
      ↓
Simulated New Dataset
```

The current dataset contains small shifts in temperature and distance.

Run:

```bash
python src/monitor.py
```

to generate the drift report.

---

## FastAPI

The trained models are exposed through a FastAPI application.

### Available Endpoints

#### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

#### Prediction

```http
POST /predict
```

The API:

1. Receives the input data.
2. Routes the input to a specialist model.
3. Predicts spoilage risk.
4. Generates risk causes.
5. Searches for a counterfactual explanation.
6. Returns the result as JSON.

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

---

## Example API Request

```json
{
  "temperature": 30,
  "humidity": 80,
  "distance_km": 100,
  "product_type": "milk",
  "initial_shelf_life": 5,
  "packaging_quality": 1
}
```

## Example API Response

```json
{
  "router_model": "fragile",
  "current_temperature": 30,
  "prediction": "High Spoilage Risk",
  "probability": 0.87,
  "causes": [
    "Temperature is high (30°C).",
    "Humidity is high (80%).",
    "Delivery distance is long (100 km).",
    "This product has a short shelf life (5 days).",
    "Packaging quality is poor."
  ],
  "counterfactual": "If temperature is reduced from 30°C to 15°C, the predicted spoilage risk decreases to Low."
}
```

---

## Project Structure

```text
grocery-spoilage-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── grocery_data.csv
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── counterfactual.py
│   ├── data_generator.py
│   ├── monitor.py
│   ├── pipeline.py
│   └── train.py
│
├── tests/
│   ├── __init__.py
│   └── test_api.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/grocery-spoilage-mlops.git
cd grocery-spoilage-mlops
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Generate the dataset

```bash
python src/data_generator.py
```

### Train the models

```bash
python src/train.py
```

This trains:

* Router model
* Hardy specialist
* Normal specialist
* Fragile specialist

The training process also records experiment metrics using MLflow.

### Run using Prefect

```bash
python src/pipeline.py
```

### Start the FastAPI server

```bash
uvicorn src.api:app --reload
```

Open:

```text
http://localhost:8000/docs
```

### Run tests

```bash
pytest -v
```

### Open MLflow

```bash
mlflow ui
```

### Generate the drift report

```bash
python src/monitor.py
```

---

## GitHub Actions

The project includes a GitHub Actions workflow:

```text
.github/workflows/ci.yml
```

The workflow runs automatically on pushes and pull requests.

It performs:

1. Repository checkout
2. Python setup
3. Dependency installation
4. Dataset generation
5. Model training
6. Pytest execution

This provides basic continuous integration and helps detect broken code before changes are merged.

---

## Model Performance

Typical results on the test set are approximately:

| Model              | Accuracy |
| ------------------ | -------: |
| Router             |    ~0.94 |
| Hardy Specialist   |    ~0.98 |
| Normal Specialist  |    ~0.89 |
| Fragile Specialist |    ~0.90 |

Because the dataset is generated randomly, results can vary slightly between runs.

These results are based on a **synthetic dataset** and should not be interpreted as real-world production performance.

---

## Limitations

* The dataset is synthetic and rule-based.
* The models have not been trained on real grocery spoilage records.
* The router does not directly use the product name.
* The router can occasionally select the wrong specialist category.
* Counterfactual search only changes a limited set of features.
* Multiple-feature counterfactual changes are not currently supported.
* Packaging quality is not included in the counterfactual search.
* There is no authentication or database.
* MLflow is configured for local experiment tracking.
* The application is intended as a local academic/demo project rather than a production service.

---

## Future Improvements

Possible improvements include:

* Use a real-world grocery spoilage dataset.
* Improve the model-routing strategy.
* Add multi-feature counterfactual explanations.
* Add Docker support.
* Add a web-based frontend.
* Add model versioning using the MLflow Model Registry.
* Add persistent monitoring and logging.
* Deploy the API to a cloud platform.

---

## Technologies Used

```text
Python
Scikit-learn
XGBoost
FastAPI
Pydantic
MLflow
Prefect
Evidently
Pytest
GitHub Actions
Pandas
NumPy
```

---

## Project Goal

The main goal of this project is to demonstrate how a complete machine-learning workflow can be combined with basic MLOps practices:

```text
Data Generation
      ↓
Model Training
      ↓
Model Routing
      ↓
Prediction
      ↓
Explainability
      ↓
Experiment Tracking
      ↓
Pipeline Orchestration
      ↓
Data Drift Monitoring
      ↓
API
      ↓
Continuous Integration
```

This makes the project suitable for demonstrating **Machine Learning + MLOps + Explainable AI concepts** in an academic project or viva.

---

## Author

**Nikhil Bhargav**

College-level MLOps project focused on machine learning model routing, explainability, and deployment workflows.
