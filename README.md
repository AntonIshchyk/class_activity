# Stroke Risk Assessment System (Class Activity)

A machine learning-powered web application for assessing patient stroke risk using a Random Forest classifier. Made by:

- Amen Tsehaie (1050045)
- Anton Ishchyk (1077546)
- Mark Salloum (1080225)
- Milan van Zijderveld (1053258)

## What It Does

The application predicts the probability that a patient will have a stroke based on seven key health indicators:

- **Age** — Patient age in years
- **Gender** — Biological sex (Male/Female)
- **Hypertension** — History of high blood pressure (Yes/No)
- **Heart Disease** — Known heart disease (Yes/No)
- **Average Glucose** — Blood glucose concentration (mg/dL)
- **BMI** — Body Mass Index
- **Smoking Status** — Current or historical smoking status

### Risk Classification

Predictions are categorized into three risk levels with clinical recommendations:

| Risk Level   | Probability | Color            | Recommendation                                        |
| ------------ | ----------- | ---------------- | ----------------------------------------------------- |
| **LOW**      | < 25%       | success (green)  | No immediate escalation. Continue standard protocol.  |
| **MODERATE** | 25% - 54%   | warning (yellow) | Recommend neurological assessment within 2 hours.     |
| **HIGH**     | ≥ 55%       | danger (red)     | Urgent — refer for immediate neurological evaluation. |

## Project Structure

```
.
├── app.py                                    # Flask web server
├── train.py                                  # Model training pipeline
├── stroke_model.pkl                          # Trained Random Forest model (binary)
├── healthcare-dataset-stroke-data.csv        # Training dataset
├── templates/
│   └── index.html                            # Web interface
└── README.md                                 # This file
```

## Files

**What each file does:**

- `train.py` — Trains a Random Forest classifier on the healthcare dataset and saves the model as a pickle file.
- `app.py` — Flask web server that serves the web interface and provides the `/predict` endpoint for stroke risk predictions.
- `healthcare-dataset-stroke-data.csv` — Contains 5,110 patient health records used for training the machine learning model.
- `templates/index.html` — HTML form providing the user interface for patients to input health data and receive risk assessments.
- `requirements.txt` — Lists Python package dependencies needed to run the application.

**Extensions and design decisions:**
The model uses a Random Forest classifier with balanced class weights to handle the imbalanced dataset (only ~5% of patients had strokes). Risk predictions are classified into three clinical categories with color-coded recommendations: LOW risk (<25%) shown in green requires no escalation, MODERATE risk (25%-54%) shown in yellow recommends neurological assessment within 2 hours, and HIGH risk (≥55%) shown in red requires urgent neurological evaluation. The application encodes categorical features (gender and smoking status) and normalizes numerical inputs before prediction, ensuring consistent model input across different user submissions. 2. **Data Cleaning** — Drops missing values 3. **Feature Encoding**:

- `gender`: Maps 'Male' → 1, 'Female' → 0, 'Other' → 0
- `smoking_status`: Maps to numeric values (0, 1, 2, -1)

4. **Model Training** — Fits RandomForestClassifier with 200 estimators, max depth 8
5. **Model Serialization** — Saves model as a pickle dictionary: `{'model': model, 'features': FEATURES}`

**Run**: `python train.py`

### `app.py` — Flask Web Server

The production web application with two routes:

#### **GET `/`** — Home Page

Renders the stroke assessment form where users input patient data.

#### **POST `/predict`** — Risk Prediction

1. **Data Extraction** — Reads form fields (age, gender, hypertension, etc.)
2. **Feature Encoding** — Converts categorical fields to numeric using the same encoding as training:
   - Gender: 'male' → 1, else → 0
   - Smoking Status: 'never smoked' → 0, 'formerly smoked' → 1, 'smokes' → 2, 'Unknown' → -1
3. **Prediction** — Calls `model.predict_proba()` to get stroke probability (0-1)
4. **Risk Mapping** — Classifies probability into LOW/MODERATE/HIGH with clinical message
5. **Response** — Renders template with prediction results and original patient data

**Run**: `python app.py` → Visit `http://localhost:5000`

### `templates/index.html` — Web Interface

Bootstrap-based responsive form with:

- Input fields for all 7 patient metrics
- Radio buttons for yes/no fields
- Dropdown for smoking status
- Results display showing risk level, probability, and clinical recommendation
- Link to original Kaggle dataset

## How It Works

### Training Pipeline (train.py)

```
healthcare-dataset-stroke-data.csv
         ↓
    Load & Clean Data
         ↓
   Encode Features
    - gender: Male→1, Female→0, Other→0
    - smoking: never→0, formerly→1, smokes→2, Unknown→-1
         ↓
   Train RandomForest (200 trees, max depth 8)
         ↓
   Evaluate & Save stroke_model.pkl
```

### Prediction Pipeline (app.py)

```
User submits form (age, gender, hypertension, etc.)
         ↓
Extract form fields with type conversion & defaults
         ↓
Encode categorical features (same as training)
         ↓
Create feature vector [age, hypertension, heart_disease, ...]
         ↓
model.predict_proba() → probability (0-1)
         ↓
Classify: < 0.25 = LOW, < 0.55 = MODERATE, ≥ 0.55 = HIGH
         ↓
Render results with patient data & clinical recommendation
```

## Requirements

- Python 3.8+
- Flask 3.1.2
- scikit-learn 1.7.2+
- pandas
- numpy
- joblib

Install dependencies:

```bash
pip install flask scikit-learn pandas numpy joblib
```

## Usage

### 1. Train the Model (One-time)

```bash
python train.py
```

This generates `stroke_model.pkl` containing the trained Random Forest.

### 2. Start the Web Server

```bash
python app.py
```

Server runs on `http://localhost:5000` with debug mode enabled.

### 3. Use the Web Interface

- Navigate to `http://localhost:5000`
- Fill in patient information
- Click "Assess Stroke Risk"
- View predicted risk level and clinical recommendation

## Data Processing

### Encoding Strategy

Categorical features are encoded consistently between training and prediction:

- **Gender**: Male = 1, Female = 0
- **Smoking Status**:
  - Never smoked = 0
  - Formerly smoked = 1
  - Smokes = 2
  - Unknown = -1

### Default Values

Form fields include sensible defaults for missing input:

- Numeric fields default to 0.0
- Categorical fields default to safe values ('female' for gender, 'Unknown' for smoking)

## Model Performance

The Random Forest classifier was trained with:

- **200 estimators** (decision trees)
- **Max depth**: 8 (prevents overfitting)
- **Class weights**: 'balanced' (handles class imbalance in stroke data)
- **Train/test split**: 80/20 with stratification
- **Evaluation**: Classification report and ROC-AUC score

## Clinical Disclaimer

⚠️ **This system is for decision support only.** Predictions should be validated by qualified healthcare professionals. This tool assists emergency department triage but does not replace comprehensive clinical evaluation.

## Dataset Source

Training data sourced from [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
