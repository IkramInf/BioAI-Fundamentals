# Complete AI Model Development & Deployment Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Phase 1: Planning & Requirements](#phase-1-planning--requirements)
3. [Phase 2: Data Preparation](#phase-2-data-preparation)
4. [Phase 3: Model Development](#phase-3-model-development)
5. [Phase 4: Model Evaluation](#phase-4-model-evaluation)
6. [Phase 5: Deployment Preparation](#phase-5-deployment-preparation)
7. [Phase 6: Production Deployment](#phase-6-production-deployment)
8. [Phase 7: Monitoring & Maintenance](#phase-7-monitoring--maintenance)
9. [Best Practices & Common Pitfalls](#best-practices--common-pitfalls)
10. [Checklists](#checklists)

---

## Introduction

This guide walks you through the complete lifecycle of developing and deploying an AI model to production. Whether you're building a recommendation system, image classifier, or natural language processing model, these principles apply universally.

**What you'll learn:**
- How to plan and scope your AI project
- Best practices for data handling and model training
- How to deploy models safely to production
- How to monitor and maintain models over time

---

## Phase 1: Planning & Requirements

### 1.1 Define the Problem

**What it means:** Before writing any code, clearly understand what business problem you're solving.

**Questions to answer:**
- What specific problem are we solving?
- Who are the end users?
- What does success look like? (Define metrics)
- What are the constraints? (latency, cost, accuracy requirements)

**Example:**
- ❌ Bad: "Build an AI to improve sales"
- ✅ Good: "Build a product recommendation model that suggests 5 relevant items to users, with at least 20% click-through rate, response time under 100ms"

### 1.2 Feasibility Assessment

**Check if AI is the right solution:**
- Is there enough data available? (typically need 1000+ examples minimum)
- Do you have labels/ground truth? (for supervised learning)
- Can the problem be solved with simpler methods first? (rule-based systems, SQL queries)
- What's the cost-benefit analysis?

### 1.3 Define Success Metrics

**Business metrics:**
- Revenue impact, user engagement, cost savings

**Technical metrics:**
- Accuracy, precision, recall, F1-score
- Latency (response time)
- Throughput (requests per second)

**Example metric definition:**
```
Primary metric: Precision @ 90% recall
Target: ≥ 85% precision
Latency requirement: < 200ms (p99)
Minimum throughput: 100 requests/second
```

### 1.4 Choose Your Tech Stack

**Framework selection:**
- **TensorFlow/Keras**: Great for production, large-scale deployments
- **PyTorch**: Excellent for research, increasingly popular in production
- **Scikit-learn**: Perfect for traditional ML (not deep learning)
- **Hugging Face Transformers**: Best for NLP tasks

**Infrastructure:**
- **Cloud providers**: AWS, Google Cloud, Azure
- **Containerization**: Docker (essential for deployment)
- **Orchestration**: Kubernetes (for scaling)
- **Model serving**: TensorFlow Serving, TorchServe, FastAPI, BentoML

---

## Phase 2: Data Preparation

### 2.1 Data Collection

**Sources to consider:**
- Existing databases and logs
- User interactions and behavior
- Third-party datasets
- Synthetic data generation
- Manual labeling services

**Important considerations:**
- **Privacy & compliance**: GDPR, CCPA, data retention policies
- **Data licensing**: Can you legally use this data?
- **Bias**: Is your data representative of all user groups?

### 2.2 Data Storage

**Options:**
- **Data lakes**: S3, Google Cloud Storage (raw data)
- **Data warehouses**: BigQuery, Snowflake, Redshift (structured data)
- **Feature stores**: Feast, Tecton (ML-specific features)
- **Version control**: DVC (Data Version Control), Git LFS

**Best practices:**
```
my-ml-project/
├── data/
│   ├── raw/              # Original, immutable data
│   ├── processed/        # Cleaned, transformed data
│   ├── features/         # Engineered features
│   └── README.md         # Data documentation
```

### 2.3 Data Exploration (EDA)

**What to check:**
```python
# Basic statistics
- Number of samples
- Number of features
- Data types
- Missing values percentage
- Class distribution (for classification)
- Statistical summary (mean, median, std)

# Visualizations
- Histograms (distribution of features)
- Correlation matrices
- Box plots (outliers)
- Class balance charts
```

**Tools:**
- Pandas, Matplotlib, Seaborn
- Jupyter notebooks for exploration

### 2.4 Data Cleaning

**Common issues and solutions:**

**Missing values:**
```python
# Options:
1. Remove rows with missing values (if < 5% of data)
2. Impute with mean/median/mode
3. Use algorithms that handle missing values (XGBoost)
4. Create a "missing" indicator feature
```

**Outliers:**
```python
# Detect using:
- IQR (Interquartile Range) method
- Z-score (statistical method)
- Domain knowledge

# Handle by:
- Removing (if data errors)
- Capping (winsorization)
- Transforming (log transform)
```

**Duplicates:**
```python
# Always check for and remove duplicate records
df.drop_duplicates(inplace=True)
```

### 2.5 Feature Engineering

**What it means:** Creating new features from existing data to help your model learn better.

**Common techniques:**

**Numerical features:**
```python
# Scaling (very important!)
- StandardScaler: (x - mean) / std
- MinMaxScaler: (x - min) / (max - min)
- RobustScaler: resistant to outliers

# Transformations
- Log transform: for skewed distributions
- Polynomial features: x² , x³
- Binning: converting continuous to categorical
```

**Categorical features:**
```python
# Encoding methods
- One-hot encoding: for nominal categories (< 10 categories)
- Label encoding: for ordinal categories
- Target encoding: mean of target per category
- Embeddings: for high-cardinality categories (neural networks)
```

**Time-based features:**
```python
# Extract from timestamps
- Day of week, month, year
- Hour of day
- Is weekend, is holiday
- Time since last event
```

**Domain-specific features:**
- Text: TF-IDF, word embeddings, sentence length
- Images: edges, colors, textures (or use CNNs)
- Time series: lags, rolling averages, trends

### 2.6 Data Splitting

**Standard split:**
```python
# Train: 70-80% (for training the model)
# Validation: 10-15% (for tuning hyperparameters)
# Test: 10-15% (for final evaluation, NEVER use during development)

from sklearn.model_selection import train_test_split

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Second split: separate train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)
# 0.176 of 85% ≈ 15% of total data
```

**Important:**
- **Time-based data**: Use chronological splits (train on past, test on future)
- **Stratification**: Maintain class distribution in splits
- **Data leakage**: Never let information from test set influence training

---

## Phase 3: Model Development

### 3.1 Baseline Model

**Why start with a baseline:**
A simple model gives you a reference point. If your complex model isn't much better, maybe you don't need complexity.

**Good baseline options:**
- **Classification**: Logistic Regression, Random Forest
- **Regression**: Linear Regression, Ridge/Lasso
- **Time series**: Moving average, simple ARIMA
- **NLP**: TF-IDF + Logistic Regression
- **Images**: Pre-trained ResNet with fine-tuning

**Example baseline workflow:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Train simple baseline
baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train, y_train)

# Evaluate
y_pred = baseline_model.predict(X_val)
print(classification_report(y_val, y_pred))

# Record results: Baseline accuracy: 75%
```

### 3.2 Model Selection

**Choose based on:**
- Problem type (classification, regression, clustering)
- Data size (deep learning needs lots of data)
- Interpretability needs (business requirement?)
- Latency requirements
- Available compute resources

**Quick selection guide:**

**Tabular data (structured):**
- Small data (< 10K rows): Linear models, Random Forest
- Medium data (10K-1M rows): XGBoost, LightGBM, CatBoost
- Large data (> 1M rows): Deep neural networks

**Images:**
- CNNs (Convolutional Neural Networks)
- Pre-trained models: ResNet, EfficientNet, Vision Transformers

**Text:**
- Traditional: TF-IDF + ML algorithms
- Modern: BERT, GPT, T5 (transformer models)

**Time series:**
- Statistical: ARIMA, SARIMA, Prophet
- ML: XGBoost with lag features
- Deep Learning: LSTM, GRU, Temporal Convolutional Networks

### 3.3 Model Training

**Setup your training pipeline:**

```python
# 1. Define model architecture
model = YourModelClass(
    input_dim=X_train.shape[1],
    hidden_layers=[128, 64, 32],
    output_dim=num_classes
)

# 2. Choose optimizer and loss function
optimizer = Adam(learning_rate=0.001)
loss_function = CrossEntropyLoss()

# 3. Training loop with validation
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_function)
    
    # Validation phase
    model.eval()
    val_loss = validate(model, val_loader, loss_function)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model(model, 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience_limit:
            print("Early stopping triggered")
            break
    
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
```

**Key concepts:**

**Learning rate:**
- Too high: model won't converge
- Too low: training takes forever
- Start with 0.001, use learning rate schedulers

**Batch size:**
- Smaller (16-64): better generalization, more memory efficient
- Larger (128-512): faster training, requires more memory
- Rule of thumb: as large as your GPU memory allows

**Epochs:**
- One epoch = one pass through entire training dataset
- Use early stopping to prevent overfitting
- Monitor validation loss, stop if it stops improving

**Regularization techniques:**
```python
# Prevent overfitting
1. Dropout: randomly disable neurons during training
2. L1/L2 regularization: penalize large weights
3. Data augmentation: create variations of training data
4. Batch normalization: normalize layer inputs
5. Early stopping: stop when validation loss stops improving
```

### 3.4 Hyperparameter Tuning

**What are hyperparameters:**
Settings you choose before training (not learned by the model).

**Common hyperparameters:**
- Learning rate
- Number of layers/neurons
- Batch size
- Dropout rate
- Regularization strength

**Tuning strategies:**

**1. Manual tuning:**
- Start with reasonable defaults
- Change one parameter at a time
- Good for understanding model behavior

**2. Grid search:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 500]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

**3. Random search:**
- Faster than grid search
- Samples random combinations
- Often finds good solutions quicker

**4. Bayesian optimization:**
- Intelligent search using past results
- Libraries: Optuna, Hyperopt
- Best for expensive-to-train models

**Pro tip:** Start with random search to find promising regions, then do focused grid search.

### 3.5 Experiment Tracking

**Why track experiments:**
You'll train dozens of models. Without tracking, you'll forget what worked.

**What to track:**
- Hyperparameters used
- Training/validation metrics
- Model architecture
- Dataset version
- Code version (git commit)
- Training time
- Hardware used

**Tools:**
- **MLflow**: Open source, popular
- **Weights & Biases (W&B)**: Great visualization
- **TensorBoard**: Good for deep learning
- **Neptune**: Team collaboration features

**Example with MLflow:**
```python
import mlflow

mlflow.start_run()
mlflow.log_param("learning_rate", 0.001)
mlflow.log_param("batch_size", 32)
mlflow.log_metric("train_accuracy", 0.92)
mlflow.log_metric("val_accuracy", 0.88)
mlflow.log_artifact("model.pth")
mlflow.end_run()
```

---

## Phase 4: Model Evaluation

### 4.1 Evaluation Metrics

**Classification metrics:**

**Accuracy:**
```
Accuracy = (Correct predictions) / (Total predictions)
```
- Simple but can be misleading with imbalanced data
- Use when classes are balanced

**Precision:**
```
Precision = True Positives / (True Positives + False Positives)
```
- "Of all positive predictions, how many were correct?"
- Use when false positives are costly (spam detection, fraud detection)

**Recall (Sensitivity):**
```
Recall = True Positives / (True Positives + False Negatives)
```
- "Of all actual positives, how many did we catch?"
- Use when false negatives are costly (disease detection, security threats)

**F1 Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Harmonic mean of precision and recall
- Good for imbalanced datasets

**Confusion Matrix:**
```
                Predicted
              Pos    Neg
Actual  Pos   TP     FN
        Neg   FP     TN
```

**ROC-AUC:**
- Area under ROC curve
- Measures model's ability to distinguish classes
- Higher is better (1.0 = perfect, 0.5 = random)

**Regression metrics:**

**MAE (Mean Absolute Error):**
```
MAE = mean(|actual - predicted|)
```
- Easy to interpret (same units as target)
- Not sensitive to outliers

**MSE (Mean Squared Error):**
```
MSE = mean((actual - predicted)²)
```
- Penalizes large errors more

**RMSE (Root Mean Squared Error):**
```
RMSE = sqrt(MSE)
```
- Same units as target variable
- More interpretable than MSE

**R² Score:**
```
R² = 1 - (MSE of model / MSE of mean baseline)
```
- Percentage of variance explained
- 1.0 = perfect, 0.0 = no better than mean

### 4.2 Cross-Validation

**Why use cross-validation:**
Single train/test split might be lucky or unlucky. Cross-validation gives more reliable estimates.

**K-Fold Cross-Validation:**
```python
from sklearn.model_selection import cross_val_score

# Split data into 5 folds
# Train on 4, test on 1, repeat 5 times
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

**When to use:**
- Small to medium datasets
- When you need robust performance estimates
- NOT for time-series data (use time-based splits instead)

### 4.3 Error Analysis

**Dive deep into mistakes:**

**1. Look at misclassified examples:**
```python
# Find where model is wrong
errors = X_test[y_pred != y_test]
error_labels = y_test[y_pred != y_test]

# Analyze patterns
# Are certain classes confused?
# Are certain features causing issues?
```

**2. Confusion between classes:**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
```

**3. Feature importance:**
```python
# For tree-based models
feature_importance = model.feature_importances_

# Plot top 20 features
import pandas as pd
feat_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(20)
```

**4. Check for bias:**
- Does model perform differently across demographics?
- Are certain groups disadvantaged?
- Use fairness metrics (equalized odds, demographic parity)

### 4.4 Model Comparison

**Compare your models systematically:**

```python
results = {
    'Baseline (Random Forest)': {
        'accuracy': 0.75,
        'precision': 0.73,
        'recall': 0.71,
        'f1': 0.72,
        'training_time': '5 min',
        'inference_time': '10ms'
    },
    'XGBoost': {
        'accuracy': 0.82,
        'precision': 0.81,
        'recall': 0.80,
        'f1': 0.805,
        'training_time': '15 min',
        'inference_time': '8ms'
    },
    'Deep Neural Network': {
        'accuracy': 0.85,
        'precision': 0.84,
        'recall': 0.83,
        'f1': 0.835,
        'training_time': '2 hours',
        'inference_time': '15ms'
    }
}
```

**Decision criteria:**
- Performance vs complexity trade-off
- Training time and cost
- Inference latency requirements
- Model interpretability needs
- Maintenance burden

**Pick the simplest model that meets requirements!**

---

## Phase 5: Deployment Preparation

### 5.1 Model Serialization

**Save your trained model:**

**PyTorch:**
```python
# Save model weights
torch.save(model.state_dict(), 'model_weights.pth')

# Save entire model
torch.save(model, 'full_model.pth')

# Load model
model = TheModelClass()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # Set to evaluation mode
```

**TensorFlow/Keras:**
```python
# Save model
model.save('my_model.h5')  # HDF5 format
model.save('my_model')     # SavedModel format (recommended)

# Load model
from tensorflow import keras
model = keras.models.load_model('my_model')
```

**Scikit-learn:**
```python
import joblib

# Save
joblib.dump(model, 'model.pkl')

# Load
model = joblib.load('model.pkl')
```

**Best practices:**
- Save preprocessing pipeline with model
- Version your models (model_v1.0.0.pkl)
- Save model metadata (date, accuracy, features used)

### 5.2 Model Optimization

**Why optimize:**
Production needs fast, efficient models. Training accuracy is useless if predictions take 10 seconds.

**Techniques:**

**1. Model pruning:**
```
Remove unnecessary neurons/weights
- Reduces model size by 50-90%
- Minimal accuracy loss
- Tools: TensorFlow Model Optimization, PyTorch pruning
```

**2. Quantization:**
```
Convert weights from float32 to int8
- 4x smaller model size
- Faster inference
- 1-2% accuracy loss typically
- Especially important for mobile/edge deployment
```

**3. Knowledge distillation:**
```
Train small "student" model to mimic large "teacher" model
- Retains most of the performance
- Much faster inference
```

**4. ONNX (Open Neural Network Exchange):**
```python
# Convert PyTorch to ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# ONNX provides:
- Framework interoperability
- Optimized inference engines
- Hardware-specific optimizations
```

**5. Batch predictions:**
```python
# Instead of processing one by one
for item in items:
    predict(item)  # Slow!

# Process in batches
predictions = predict(items)  # Much faster!
```

### 5.3 Create Inference Pipeline

**Package everything needed for predictions:**

```python
# inference.py
import joblib
import numpy as np

class ModelPipeline:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
    def preprocess(self, raw_input):
        """Convert raw input to model features"""
        # Handle missing values
        # Scale features
        # Encode categories
        features = self.scaler.transform(raw_input)
        return features
    
    def predict(self, raw_input):
        """End-to-end prediction"""
        features = self.preprocess(raw_input)
        predictions = self.model.predict(features)
        return self.postprocess(predictions)
    
    def postprocess(self, predictions):
        """Convert model output to user-friendly format"""
        # Convert probabilities to labels
        # Add confidence scores
        # Format output
        return predictions

# Usage
pipeline = ModelPipeline('model.pkl', 'scaler.pkl')
result = pipeline.predict(user_input)
```

### 5.4 API Development

**Create a REST API for your model:**

**Using FastAPI (recommended):**

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="ML Model API")

# Load model at startup
pipeline = ModelPipeline('model.pkl', 'scaler.pkl')

class PredictionRequest(BaseModel):
    """Define input schema"""
    feature1: float
    feature2: str
    feature3: int

class PredictionResponse(BaseModel):
    """Define output schema"""
    prediction: str
    confidence: float
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Prediction endpoint"""
    try:
        # Convert request to model input
        input_data = request.dict()
        
        # Make prediction
        result = pipeline.predict(input_data)
        
        return PredictionResponse(
            prediction=result['label'],
            confidence=result['confidence'],
            model_version="1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Test your API locally:**
```bash
# Start server
python app.py

# Test with curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.5, "feature2": "category_a", "feature3": 42}'
```

### 5.5 Containerization with Docker

**Why Docker:**
- Ensures consistency across environments
- Packages all dependencies
- Easy to deploy anywhere

**Create Dockerfile:**

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model.pkl scaler.pkl ./
COPY app.py inference.py ./

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**requirements.txt:**
```
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
scikit-learn==1.3.2
numpy==1.24.3
joblib==1.3.2
```

**Build and run Docker container:**
```bash
# Build image
docker build -t ml-model-api:v1.0.0 .

# Run container
docker run -p 8000:8000 ml-model-api:v1.0.0

# Test
curl http://localhost:8000/health
```

**.dockerignore:**
```
__pycache__
*.pyc
.git
.env
notebooks/
tests/
*.ipynb
data/raw/
```

### 5.6 Testing Before Deployment

**Unit tests for your model:**

```python
# test_model.py
import pytest
from inference import ModelPipeline

@pytest.fixture
def pipeline():
    return ModelPipeline('model.pkl', 'scaler.pkl')

def test_prediction_shape(pipeline):
    """Test output shape"""
    input_data = {'feature1': 1.0, 'feature2': 'A', 'feature3': 10}
    result = pipeline.predict(input_data)
    assert 'prediction' in result
    assert 'confidence' in result

def test_prediction_range(pipeline):
    """Test prediction is in valid range"""
    input_data = {'feature1': 1.0, 'feature2': 'A', 'feature3': 10}
    result = pipeline.predict(input_data)
    assert 0 <= result['confidence'] <= 1

def test_handles_missing_values(pipeline):
    """Test graceful handling of edge cases"""
    input_data = {'feature1': None, 'feature2': 'A', 'feature3': 10}
    # Should not crash
    result = pipeline.predict(input_data)

def test_inference_speed(pipeline):
    """Test prediction latency"""
    import time
    input_data = {'feature1': 1.0, 'feature2': 'A', 'feature3': 10}
    
    start = time.time()
    pipeline.predict(input_data)
    duration = time.time() - start
    
    assert duration < 0.1  # Should be under 100ms
```

**Integration tests for API:**

```python
# test_api.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint():
    payload = {
        "feature1": 1.5,
        "feature2": "category_a",
        "feature3": 42
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_invalid_input():
    payload = {"invalid": "data"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error
```

**Run tests:**
```bash
pytest test_model.py test_api.py -v
```

---

## Phase 6: Production Deployment

### 6.1 Choose Deployment Strategy

**Options:**

**1. Cloud VM (EC2, GCE, Azure VM):**
- **Pros**: Full control, easy to start
- **Cons**: You manage everything (scaling, updates, security)
- **Good for**: Small-medium scale, learning

**2. Kubernetes:**
- **Pros**: Auto-scaling, high availability, industry standard
- **Cons**: Complex, steep learning curve
- **Good for**: Large scale, enterprise

**3. Serverless (AWS Lambda, Google Cloud Functions):**
- **Pros**: Zero server management, pay per use
- **Cons**: Cold start latency, resource limits
- **Good for**: Sporadic traffic, small models

**4. Managed ML platforms:**
- **AWS SageMaker**: Full ML lifecycle
- **Google Vertex AI**: GCP's ML platform
- **Azure ML**: Microsoft's solution
- **Pros**: Built-in monitoring, easy scaling
- **Cons**: Vendor lock-in, can be expensive

**5. Model serving platforms:**
- **TensorFlow Serving**: For TensorFlow models
- **TorchServe**: For PyTorch models
- **BentoML**: Framework-agnostic
- **Seldon Core**: Kubernetes-native

### 6.2 Deployment to Cloud VM (Simple Approach)

**Step-by-step AWS EC2 deployment:**

**1. Launch EC2 instance:**
```bash
# Choose:
- Ubuntu 22.04 LTS
- t3.medium (2 vCPU, 4GB RAM) - adjust based on needs
- Open port 8000 in security group
```

**2. Connect and setup:**
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
```

**3. Deploy your containerized model:**
```bash
# Copy Docker image (option 1: push to registry)
docker tag ml-model-api:v1.0.0 your-registry/ml-model-api:v1.0.0
docker push your-registry/ml-model-api:v1.0.0

# On EC2: pull and run
docker pull your-registry/ml-model-api:v1.0.0
docker run -d -p 8000:8000 --name ml-api \
  --restart unless-stopped \
  your-registry/ml-model-api:v1.0.0
```

**4. Setup reverse proxy with Nginx:**
```bash
# Install Nginx
sudo apt install nginx -y

# Configure Nginx
sudo nano /etc/nginx/sites-available/ml-api

# Add configuration:
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/ml-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**5. Setup SSL (HTTPS):**
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

### 6.3 Deployment to Kubernetes

**Why Kubernetes:**
- Automatic scaling based on load
- Self-healing (restarts failed containers)
- Rolling updates with zero downtime
- Load balancing built-in

**Basic Kubernetes deployment:**

**1. Create deployment.yaml:**
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
  labels:
    app: ml-model
spec:
  replicas: 3  # Run 3 instances
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: ml-model
        image: your-registry/ml-model-api:v1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

**2. Deploy to Kubernetes:**
```bash
# Apply configuration
kubectl apply -f deployment.yaml

# Check status
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/ml-model-deployment

# Scale deployment
kubectl scale deployment ml-model-deployment --replicas=5
```

**3. Setup autoscaling:**
```yaml
# hpa.yaml (Horizontal Pod Autoscaler)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

```bash
kubectl apply -f hpa.yaml
```

### 6.4 CI/CD Pipeline

**Automate deployment with CI/CD:**

**Example with GitHub Actions:**

```yaml
# .github/workflows/deploy.yml
name: Deploy ML Model

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    
    - name: Run tests
      run: pytest tests/ -v
  
  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t ml-model-api:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
        docker tag ml-model-api:${{ github.sha }} your-registry/ml-model-api:${{ github.sha }}
        docker tag ml-model-api:${{ github.sha }} your-registry/ml-model-api:latest
        docker push your-registry/ml-model-api:${{ github.sha }}
        docker push your-registry/ml-model-api:latest
  
  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    
    steps:
    - name: Deploy to Kubernetes
      run: |
        # Update Kubernetes deployment with new image
        kubectl set image deployment/ml-model-deployment \
          ml-model=your-registry/ml-model-api:${{ github.sha }}
```

**CI/CD best practices:**
- Always run tests before deployment
- Use semantic versioning (v1.0.0, v1.0.1)
- Deploy to staging environment first
- Implement rollback mechanisms
- Tag Docker images with git commit SHA

### 6.5 Blue-Green Deployment

**Zero-downtime deployments:**

**Strategy:**
- Run two identical environments (Blue = current, Green = new)
- Deploy new version to Green
- Test Green thoroughly
- Switch traffic from Blue to Green
- Keep Blue as instant rollback option

**Implementation in Kubernetes:**

```yaml
# blue-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
      version: blue
  template:
    metadata:
      labels:
        app: ml-model
        version: blue
    spec:
      containers:
      - name: ml-model
        image: your-registry/ml-model-api:v1.0.0
---
# green-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
      version: green
  template:
    metadata:
      labels:
        app: ml-model
        version: green
    spec:
      containers:
      - name: ml-model
        image: your-registry/ml-model-api:v2.0.0
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
    version: blue  # Change to 'green' to switch traffic
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

**Switch traffic:**
```bash
# Update service to point to green
kubectl patch service ml-model-service -p '{"spec":{"selector":{"version":"green"}}}'

# If issues, instantly rollback
kubectl patch service ml-model-service -p '{"spec":{"selector":{"version":"blue"}}}'
```

### 6.6 Canary Deployment

**Gradual rollout strategy:**
- Deploy new version to small subset of users (5-10%)
- Monitor metrics closely
- Gradually increase traffic if all looks good
- Full rollback if issues detected

**Implementation with Istio/Service Mesh:**

```yaml
# virtual-service.yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: ml-model
spec:
  hosts:
  - ml-model-service
  http:
  - match:
    - headers:
        user-type:
          exact: beta
    route:
    - destination:
        host: ml-model-service
        subset: v2
  - route:
    - destination:
        host: ml-model-service
        subset: v1
      weight: 90
    - destination:
        host: ml-model-service
        subset: v2
      weight: 10  # 10% to new version
```

---

## Phase 7: Monitoring & Maintenance

### 7.1 Logging

**What to log:**

**Application logs:**
```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(request: PredictionRequest):
    request_id = str(uuid.uuid4())
    
    logger.info(f"Request {request_id}: Received prediction request")
    
    try:
        start_time = time.time()
        result = pipeline.predict(request.dict())
        duration = time.time() - start_time
        
        logger.info(f"Request {request_id}: Prediction successful in {duration:.3f}s")
        logger.info(f"Request {request_id}: Result={result}")
        
        return result
    except Exception as e:
        logger.error(f"Request {request_id}: Prediction failed - {str(e)}")
        raise
```

**Important events to log:**
- Every prediction request and response
- Prediction confidence scores
- Inference latency
- Errors and exceptions
- Input data statistics
- Model version used

**Centralized logging with ELK Stack:**
```
Elasticsearch: Store logs
Logstash: Process and forward logs
Kibana: Visualize and search logs
```

**Or use cloud solutions:**
- AWS CloudWatch
- Google Cloud Logging
- Azure Monitor
- Datadog
- Splunk

### 7.2 Metrics & Monitoring

**Key metrics to track:**

**System metrics:**
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')
active_requests = Gauge('active_requests', 'Number of active requests')
error_counter = Counter('prediction_errors_total', 'Total prediction errors')

@app.post("/predict")
async def predict(request: PredictionRequest):
    active_requests.inc()
    prediction_counter.inc()
    
    try:
        with prediction_duration.time():
            result = pipeline.predict(request.dict())
        return result
    except Exception as e:
        error_counter.inc()
        raise
    finally:
        active_requests.dec()
```

**Infrastructure metrics:**
- CPU usage
- Memory usage
- Disk I/O
- Network throughput
- Request rate (requests per second)
- Error rate
- Latency (p50, p90, p99)

**ML-specific metrics:**
- Prediction distribution (are predictions shifting?)
- Confidence score distribution
- Feature distributions (input data drift)
- Model accuracy over time

**Setup Prometheus + Grafana:**

```yaml
# docker-compose.yml
version: '3'
services:
  ml-api:
    image: ml-model-api:latest
    ports:
      - "8000:8000"
  
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

**prometheus.yml:**
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-api'
    static_configs:
      - targets: ['ml-api:8000']
```

### 7.3 Alerting

**Setup alerts for critical issues:**

**Example alert rules (Prometheus):**

```yaml
# alerts.yml
groups:
- name: ml_model_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(prediction_errors_total[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors/sec"
  
  - alert: HighLatency
    expr: histogram_quantile(0.99, prediction_duration_seconds) > 1.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High prediction latency"
      description: "P99 latency is {{ $value }} seconds"
  
  - alert: LowConfidence
    expr: avg(prediction_confidence) < 0.5
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Model confidence dropping"
      description: "Average confidence is {{ $value }}"
  
  - alert: ServiceDown
    expr: up{job="ml-api"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "ML API is down"
```

**Alert channels:**
- Email
- Slack/Teams
- PagerDuty (for on-call rotation)
- SMS (for critical alerts)

**Alert best practices:**
- Don't alert on everything (alert fatigue)
- Set appropriate thresholds
- Include actionable information
- Have clear escalation procedures

### 7.4 Model Performance Monitoring

**Detect model degradation:**

**1. Track prediction distribution:**
```python
from collections import Counter

# Store predictions over time
prediction_tracker = []

@app.post("/predict")
async def predict(request: PredictionRequest):
    result = pipeline.predict(request.dict())
    
    # Track prediction
    prediction_tracker.append({
        'timestamp': datetime.now(),
        'prediction': result['prediction'],
        'confidence': result['confidence']
    })
    
    # Alert if distribution changes significantly
    if len(prediction_tracker) > 1000:
        check_distribution_drift(prediction_tracker)
    
    return result
```

**2. Monitor input data drift:**
```python
import numpy as np
from scipy import stats

def check_feature_drift(current_features, reference_features):
    """Detect if input features have drifted from training distribution"""
    
    for feature_name in current_features.columns:
        current = current_features[feature_name]
        reference = reference_features[feature_name]
        
        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(current, reference)
        
        if p_value < 0.05:  # Significant drift detected
            logger.warning(f"Feature drift detected in {feature_name}")
            # Trigger alert
            send_alert(f"Data drift in feature: {feature_name}")
```

**3. Collect ground truth for validation:**
```python
# Store predictions and later collect actual outcomes
prediction_store = []

@app.post("/predict")
async def predict(request: PredictionRequest):
    result = pipeline.predict(request.dict())
    
    # Store for later validation
    prediction_store.append({
        'id': str(uuid.uuid4()),
        'input': request.dict(),
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'timestamp': datetime.now()
    })
    
    return result

@app.post("/feedback")
async def record_feedback(prediction_id: str, actual_outcome: str):
    """Collect actual outcomes to measure real accuracy"""
    # Update prediction record with ground truth
    # Calculate actual model performance
    # Alert if performance degrades
```

**4. A/B testing for model updates:**
```python
import random

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Randomly assign to model A or B
    model_version = 'A' if random.random() < 0.5 else 'B'
    
    if model_version == 'A':
        result = model_a.predict(request.dict())
    else:
        result = model_b.predict(request.dict())
    
    # Track which model was used
    log_ab_test(request, result, model_version)
    
    return result
```

### 7.5 Model Retraining

**When to retrain:**
- Performance degradation detected
- Significant data drift
- New data available
- Business requirements change
- Scheduled periodic retraining (e.g., monthly)

**Automated retraining pipeline:**

```python
# retrain_pipeline.py
def retrain_model():
    """Automated retraining workflow"""
    
    # 1. Fetch new data
    new_data = fetch_data_since_last_training()
    logger.info(f"Fetched {len(new_data)} new samples")
    
    # 2. Validate data quality
    if not validate_data_quality(new_data):
        logger.error("Data quality check failed")
        return
    
    # 3. Combine with existing training data
    training_data = combine_datasets(old_data, new_data)
    
    # 4. Train new model
    logger.info("Starting model training")
    new_model = train_model(training_data)
    
    # 5. Evaluate on hold-out test set
    metrics = evaluate_model(new_model, test_data)
    logger.info(f"New model metrics: {metrics}")
    
    # 6. Compare with current production model
    if metrics['accuracy'] > current_model_accuracy * 0.95:
        # New model is at least 95% as good
        logger.info("New model approved")
        
        # 7. Save new model
        save_model(new_model, f"model_v{version}.pkl")
        
        # 8. Deploy to staging
        deploy_to_staging(new_model)
        
        # 9. Run automated tests
        if run_integration_tests():
            # 10. Deploy to production (canary)
            deploy_to_production(new_model, canary=True)
        else:
            logger.error("Integration tests failed")
    else:
        logger.warning("New model doesn't meet quality threshold")

# Schedule retraining
# Using cron or orchestration tools like Airflow
```

**Apache Airflow for orchestration:**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_retraining',
    default_args=default_args,
    description='Automated model retraining pipeline',
    schedule_interval='@weekly',  # Run weekly
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

fetch_data_task = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate',
    python_callable=evaluate_model,
    dag=dag,
)

deploy_task = PythonOperator(
    task_id='deploy',
    python_callable=deploy_to_staging,
    dag=dag,
)

# Define task dependencies
fetch_data_task >> train_model_task >> evaluate_task >> deploy_task
```

### 7.6 Documentation

**Maintain comprehensive documentation:**

**Model card:**
```markdown
# Model Card: Product Recommendation Model v2.1

## Model Details
- **Model type**: XGBoost Classifier
- **Training date**: 2024-10-15
- **Author**: ML Team
- **Version**: 2.1.0

## Intended Use
- **Primary use**: Recommend products to users on e-commerce platform
- **Out-of-scope**: Medical advice, financial decisions

## Training Data
- **Source**: User interaction logs from 2023-2024
- **Size**: 5 million samples
- **Features**: 45 features (user demographics, behavior, product attributes)

## Performance
- **Accuracy**: 87.3% on test set
- **Precision**: 85.1%
- **Recall**: 82.4%
- **Latency**: 45ms (p99)

## Limitations
- Performance drops for users with < 5 interactions
- May not generalize to new product categories
- Trained primarily on US user data

## Bias Considerations
- Tested across age groups: no significant performance differences
- Geographic bias: lower accuracy for non-US users

## Ethical Considerations
- Does not use protected attributes (race, religion, etc.)
- Regular audits for fairness

## Maintenance
- Retrained monthly
- Contact: ml-team@company.com
```

**API documentation:**
- Use OpenAPI/Swagger for REST APIs
- Include example requests and responses
- Document error codes
- Specify rate limits

**Runbook for on-call:**
```markdown
# ML Model On-Call Runbook

## Common Issues

### High Error Rate
**Symptoms**: Error rate > 5%
**Possible causes**: 
- Invalid input data
- Model file corruption
- Dependency issues

**Steps**:
1. Check application logs: `kubectl logs deployment/ml-model`
2. Verify model file integrity
3. Check input data format
4. Rollback to previous version if needed: `kubectl rollout undo deployment/ml-model`

### High Latency
**Symptoms**: p99 latency > 1s
**Possible causes**:
- High traffic
- Resource constraints
- Database slowdown

**Steps**:
1. Check CPU/memory: `kubectl top pods`
2. Scale up if needed: `kubectl scale deployment/ml-model --replicas=10`
3. Check database performance
4. Enable request caching if applicable

### Model Drift Detected
**Symptoms**: Accuracy dropping, confidence scores low
**Steps**:
1. Investigate data distribution changes
2. Check for upstream data pipeline issues
3. Trigger retraining pipeline
4. If severe, consider temporary rollback

## Contacts
- ML Team Lead: @ml-lead (Slack)
- DevOps: @devops-oncall
- PagerDuty: ml-model-alerts
```

---

## Best Practices & Common Pitfalls

### Best Practices

**1. Start Simple**
- Begin with simplest model that could work
- Add complexity only when needed
- Measure everything before optimizing

**2. Version Everything**
- Code (git)
- Data (DVC)
- Models (MLflow, model registry)
- Dependencies (requirements.txt with pinned versions)

**3. Automate Early**
- Automated testing
- CI/CD pipelines
- Automated retraining
- Automated monitoring

**4. Think About Maintenance**
- Will someone understand this in 6 months?
- Is it easy to retrain?
- Can you rollback quickly?
- Is it documented?

**5. Monitor Continuously**
- Don't deploy and forget
- Watch for model degradation
- Track data drift
- Measure business impact

**6. Security**
```python
# Input validation
from pydantic import BaseModel, validator

class PredictionRequest(BaseModel):
    value: float
    
    @validator('value')
    def validate_value(cls, v):
        if v < 0 or v > 1000:
            raise ValueError('Value must be between 0 and 1000')
        return v

# Rate limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(request: PredictionRequest):
    ...

# Authentication
from fastapi.security import HTTPBearer
security = HTTPBearer()

@app.post("/predict")
async def predict(request: PredictionRequest, token: str = Depends(security)):
    verify_token(token)
    ...
```

### Common Pitfalls

**1. Data Leakage**
```python
# ❌ BAD: Scaling before split
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# ✅ GOOD: Fit on train only
X_train, X_test = train_test_split(X)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**2. Not Testing on Representative Data**
- Test set must represent production distribution
- Watch for temporal effects (train on old, test on recent)
- Consider different user segments

**3. Ignoring Model Limitations**
- Every model has failure modes
- Document what your model can't do
- Add guardrails for edge cases

**4. Over-Engineering**
- Don't use deep learning for simple problems
- Don't build Kubernetes if cloud VM suffices
- Optimize for development speed initially

**5. Lack of Monitoring**
- Silent degradation is common
- Models drift over time
- Monitor business metrics, not just technical metrics

**6. Training-Serving Skew**
```python
# Ensure preprocessing is identical in training and serving
# ❌ BAD: Different preprocessing code
# training.py: custom preprocessing
# serving.py: different custom preprocessing

# ✅ GOOD: Shared preprocessing pipeline
# preprocessing.py: SingleSource of truth
# Used in both training and serving
```

**7. Not Having Rollback Plan**
- Always be able to quickly revert
- Keep previous model version deployed
- Test rollback procedure

**8. Treating ML Like Traditional Software**
- ML systems degrade over time (code doesn't)
- Need continuous monitoring
- Need retraining infrastructure
- Need data pipeline reliability

---

## Checklists

### Pre-Deployment Checklist

```
□ Model performance meets requirements
  □ Accuracy >= target
  □ Latency <= requirement
  □ Works on all data segments

□ Testing completed
  □ Unit tests pass
  □ Integration tests pass
  □ Load testing done
  □ Security testing done

□ Documentation complete
  □ Model card created
  □ API documentation written
  □ Runbook prepared
  □ Architecture documented

□ Infrastructure ready
  □ Deployment environment provisioned
  □ Monitoring configured
  □ Alerting setup
  □ Logging configured
  □ Backup strategy defined

□ Security measures
  □ Input validation implemented
  □ Rate limiting configured
  □ Authentication/authorization setup
  □ Secrets management configured
  □ HTTPS enabled

□ Team prepared
  □ On-call rotation defined
  □ Team trained on system
  □ Incident response plan ready
  □ Rollback procedure documented

□ Stakeholder alignment
  □ Business team informed
  □ Expected metrics communicated
  □ Limitations explained
  □ Launch plan agreed
```

### Production Launch Checklist

```
□ Pre-launch
  □ Deploy to staging
  □ Run smoke tests
  □ Verify monitoring/alerting
  □ Confirm rollback plan

□ Launch
  □ Deploy with canary (10% traffic)
  □ Monitor for 1 hour
  □ Gradually increase to 50%
  □ Monitor for 2 hours
  □ Full rollout (100%)

□ Post-launch (first 24 hours)
  □ Monitor error rates
  □ Check latency metrics
  □ Verify prediction distribution
  □ Collect user feedback
  □ Document any issues

□ Post-launch (first week)
  □ Analyze business metrics
  □ Compare to baseline
  □ Gather stakeholder feedback
  □ Plan iterations
```

### Monthly Maintenance Checklist

```
□ Performance review
  □ Check model accuracy trend
  □ Review error rates
  □ Analyze latency metrics
  □ Check resource utilization

□ Data quality
  □ Inspect input data distribution
  □ Check for data drift
  □ Validate data pipeline health

□ System health
  □ Review logs for warnings
  □ Check for security vulnerabilities
  □ Update dependencies
  □ Test backup/restore

□ Documentation
  □ Update model card if needed
  □ Review and update runbook
  □ Document new issues found

□ Planning
  □ Evaluate need for retraining
  □ Plan improvements
  □ Review technical debt
```

---

## Conclusion

Deploying an AI model to production is a journey that extends far beyond training a model. Success requires:

- **Careful planning**: Understanding the problem and setting clear success criteria
- **Robust development**: Following best practices for data handling and model training
- **Thoughtful deployment**: Choosing the right infrastructure and deployment strategy
- **Continuous monitoring**: Watching for issues and model degradation
- **Regular maintenance**: Retraining and updating as needed

Remember: **A model in production that works reliably 80% of the time is more valuable than a model that's 95% accurate in a notebook but never deployed.**

Start simple, iterate quickly, and always prioritize reliability and maintainability over complexity.

Good luck with your deployment!

---

## Additional Resources

**Learning Resources:**
- [Google's MLOps Course](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [AWS Machine Learning Best Practices](https://aws.amazon.com/machine-learning/)
- [Designing Machine Learning Systems (book by Chip Huyen)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)

**Tools Documentation:**
- [FastAPI](https://fastapi.tiangolo.com/)
- [Docker](https://docs.docker.com/)
- [Kubernetes](https://kubernetes.io/docs/)
- [MLflow](https://mlflow.org/docs/latest/index.html)
- [Prometheus](https://prometheus.io/docs/)

**Communities:**
- r/MachineLearning (Reddit)
- MLOps Community (Slack)
- Papers With Code
- Kaggle Forums
