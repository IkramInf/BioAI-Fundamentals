# Complete ML Engineer Interview Documentation

## Table of Contents
1. [Python Proficiency](#1-python-proficiency)
2. [Clean Code & Design Principles](#2-clean-code--design-principles)
3. [Containerization (Docker)](#3-containerization-docker)
4. [Deep Learning & SOTA Implementation](#4-deep-learning--sota-implementation)
5. [Data Drift & Model Drift](#5-data-drift--model-drift)
6. [Application Architecture & Integration](#6-application-architecture--integration)
7. [Orchestration Frameworks](#7-orchestration-frameworks)

---

## 1. Python Proficiency

### Q1.1: Explain Python's GIL (Global Interpreter Lock) and its implications for multi-threading.

**Answer:**
The GIL is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously. 

**Analogy:** Think of the GIL as a single microphone in a conference room. Even if multiple people (threads) want to speak (execute code), only one person can hold the microphone at a time.

**Implications:**
- CPU-bound tasks don't benefit from multi-threading
- I/O-bound tasks can still benefit (thread releases GIL during I/O)
- Use `multiprocessing` for CPU-bound parallelism

**Example:**
```python
import threading
import multiprocessing
import time

# CPU-bound task - threading won't help much due to GIL
def cpu_bound_task(n):
    return sum(i * i for i in range(n))

# Multi-threading (limited by GIL)
def with_threading():
    threads = []
    for _ in range(4):
        t = threading.Thread(target=cpu_bound_task, args=(10000000,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

# Multi-processing (bypasses GIL)
def with_multiprocessing():
    processes = []
    for _ in range(4):
        p = multiprocessing.Process(target=cpu_bound_task, args=(10000000,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
```

### Q1.2: What are Python decorators and how would you implement a retry decorator?

**Answer:**
Decorators are functions that modify the behavior of other functions or methods without changing their source code.

**Analogy:** A decorator is like gift wrapping. The gift (function) remains the same, but you add an outer layer (additional functionality) around it.

**Example:**
```python
import time
import functools
from typing import Callable, Any

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    wait_time = delay * (2 ** (attempts - 1))
                    print(f"Attempt {attempts} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2.0)
def unstable_api_call(url: str):
    """Simulates an unstable API call"""
    import random
    if random.random() < 0.7:
        raise ConnectionError("API temporarily unavailable")
    return {"status": "success", "data": "response"}
```

### Q1.3: Explain Python's memory management and garbage collection.

**Answer:**
Python uses reference counting and generational garbage collection.

**Key concepts:**
- **Reference Counting:** Each object tracks how many references point to it
- **Garbage Collection:** Detects and cleans circular references
- **Generations:** Objects are categorized into 3 generations (0, 1, 2)

**Example:**
```python
import sys
import gc

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

# Reference counting
x = Node(1)
print(sys.getrefcount(x))  # 2 (x + temporary reference in getrefcount)

y = x
print(sys.getrefcount(x))  # 3

# Circular reference - needs GC
node1 = Node(1)
node2 = Node(2)
node1.next = node2
node2.next = node1  # Circular reference

del node1, node2  # Reference count doesn't reach 0
gc.collect()  # Garbage collector cleans up circular references
```

### Q1.4: What are context managers and how do you create custom ones?

**Answer:**
Context managers handle resource management (setup/teardown) using `with` statement.

**Analogy:** Like checking into and out of a hotel - automatic setup when you arrive and cleanup when you leave.

**Example:**
```python
from contextlib import contextmanager
import time

# Method 1: Using __enter__ and __exit__
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.elapsed = self.end - self.start
        print(f"Elapsed time: {self.elapsed:.4f} seconds")
        return False  # Don't suppress exceptions

# Method 2: Using contextmanager decorator
@contextmanager
def database_connection(db_url: str):
    """Simulates database connection management"""
    conn = f"Connection to {db_url}"
    print(f"Opening {conn}")
    try:
        yield conn
    finally:
        print(f"Closing {conn}")

# Usage
with Timer():
    time.sleep(1)
    print("Processing...")

with database_connection("postgresql://localhost") as conn:
    print(f"Using {conn}")
```

### Q1.5: Explain generators and their advantages. When would you use them?

**Answer:**
Generators are functions that yield values one at a time, creating an iterator lazily.

**Advantages:**
- Memory efficient (don't load all data at once)
- Can represent infinite sequences
- Better performance for large datasets

**Analogy:** Like a ticket dispenser - produces one ticket at a time rather than printing all tickets upfront.

**Example:**
```python
import sys

# Regular function - loads all in memory
def get_numbers_list(n):
    return [i ** 2 for i in range(n)]

# Generator - produces values on demand
def get_numbers_generator(n):
    for i in range(n):
        yield i ** 2

# Memory comparison
n = 1000000
list_version = get_numbers_list(n)
gen_version = get_numbers_generator(n)

print(f"List size: {sys.getsizeof(list_version)} bytes")
print(f"Generator size: {sys.getsizeof(gen_version)} bytes")

# Practical example: Reading large files
def read_large_file(filepath):
    """Memory-efficient file reading"""
    with open(filepath, 'r') as f:
        for line in f:
            yield line.strip()

# Pipeline example
def process_data_pipeline(filepath):
    """Generator pipeline for data processing"""
    for line in read_large_file(filepath):
        # Filter
        if not line.startswith('#'):
            # Transform
            processed = line.upper()
            yield processed
```

---

## 2. Clean Code & Design Principles

### Q2.1: Explain the SOLID principles with Python examples.

**Answer:**

#### **S - Single Responsibility Principle**
A class should have only one reason to change.

**Analogy:** A chef should cook, not also clean and serve. Each role has one responsibility.

**Example:**
```python
# Bad - Multiple responsibilities
class UserManager:
    def create_user(self, name, email):
        # Database logic
        pass
    
    def send_welcome_email(self, email):
        # Email logic
        pass
    
    def generate_report(self):
        # Reporting logic
        pass

# Good - Single responsibility
class UserRepository:
    """Handles user data persistence"""
    def create_user(self, name: str, email: str):
        # Database logic only
        pass
    
    def get_user(self, user_id: int):
        pass

class EmailService:
    """Handles email operations"""
    def send_welcome_email(self, email: str):
        # Email logic only
        pass

class UserReportGenerator:
    """Handles user reporting"""
    def generate_report(self):
        # Reporting logic only
        pass
```

#### **O - Open/Closed Principle**
Open for extension, closed for modification.

**Example:**
```python
from abc import ABC, abstractmethod
from typing import List

# Bad - Need to modify class for new shapes
class AreaCalculator:
    def calculate_area(self, shapes):
        total = 0
        for shape in shapes:
            if shape['type'] == 'circle':
                total += 3.14 * shape['radius'] ** 2
            elif shape['type'] == 'rectangle':
                total += shape['width'] * shape['height']
        return total

# Good - Extensible without modification
class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius
    
    def area(self) -> float:
        return 3.14159 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def area(self) -> float:
        return self.width * self.height

class Triangle(Shape):
    def __init__(self, base: float, height: float):
        self.base = base
        self.height = height
    
    def area(self) -> float:
        return 0.5 * self.base * self.height

class AreaCalculator:
    def calculate_total_area(self, shapes: List[Shape]) -> float:
        return sum(shape.area() for shape in shapes)

# Usage - can add new shapes without modifying AreaCalculator
shapes = [Circle(5), Rectangle(4, 6), Triangle(3, 4)]
calculator = AreaCalculator()
print(calculator.calculate_total_area(shapes))
```

#### **L - Liskov Substitution Principle**
Subtypes must be substitutable for their base types.

**Example:**
```python
# Bad - Violates LSP
class Bird:
    def fly(self):
        return "Flying"

class Penguin(Bird):
    def fly(self):
        raise Exception("Penguins can't fly!")  # Breaks contract

# Good - Respects LSP
class Bird:
    def move(self):
        pass

class FlyingBird(Bird):
    def move(self):
        return self.fly()
    
    def fly(self):
        return "Flying through the air"

class Penguin(Bird):
    def move(self):
        return self.swim()
    
    def swim(self):
        return "Swimming in water"

class Sparrow(FlyingBird):
    pass

def make_bird_move(bird: Bird):
    print(bird.move())

# Works with any Bird subtype
make_bird_move(Sparrow())
make_bird_move(Penguin())
```

#### **I - Interface Segregation Principle**
Clients shouldn't depend on interfaces they don't use.

**Example:**
```python
from abc import ABC, abstractmethod

# Bad - Fat interface
class Worker(ABC):
    @abstractmethod
    def work(self):
        pass
    
    @abstractmethod
    def eat(self):
        pass

class Human(Worker):
    def work(self):
        return "Working"
    
    def eat(self):
        return "Eating lunch"

class Robot(Worker):
    def work(self):
        return "Working"
    
    def eat(self):
        raise NotImplementedError("Robots don't eat!")

# Good - Segregated interfaces
class Workable(ABC):
    @abstractmethod
    def work(self):
        pass

class Eatable(ABC):
    @abstractmethod
    def eat(self):
        pass

class Human(Workable, Eatable):
    def work(self):
        return "Working"
    
    def eat(self):
        return "Eating lunch"

class Robot(Workable):
    def work(self):
        return "Working"
```

#### **D - Dependency Inversion Principle**
Depend on abstractions, not concretions.

**Example:**
```python
from abc import ABC, abstractmethod

# Bad - High-level depends on low-level
class MySQLDatabase:
    def save(self, data):
        print(f"Saving {data} to MySQL")

class UserService:
    def __init__(self):
        self.db = MySQLDatabase()  # Tight coupling
    
    def save_user(self, user):
        self.db.save(user)

# Good - Both depend on abstraction
class Database(ABC):
    @abstractmethod
    def save(self, data):
        pass

class MySQLDatabase(Database):
    def save(self, data):
        print(f"Saving {data} to MySQL")

class PostgreSQLDatabase(Database):
    def save(self, data):
        print(f"Saving {data} to PostgreSQL")

class MongoDatabase(Database):
    def save(self, data):
        print(f"Saving {data} to MongoDB")

class UserService:
    def __init__(self, database: Database):
        self.db = database  # Depends on abstraction
    
    def save_user(self, user):
        self.db.save(user)

# Easy to swap implementations
user_service = UserService(MySQLDatabase())
user_service = UserService(PostgreSQLDatabase())
```

### Q2.2: Explain DRY (Don't Repeat Yourself) principle with examples.

**Answer:**
DRY means avoiding code duplication by abstracting common patterns.

**Analogy:** Like creating a recipe card instead of writing the same cooking instructions multiple times.

**Example:**
```python
# Bad - Repetitive code
def calculate_discount_bronze(price):
    tax = price * 0.1
    discount = price * 0.05
    return price + tax - discount

def calculate_discount_silver(price):
    tax = price * 0.1
    discount = price * 0.10
    return price + tax - discount

def calculate_discount_gold(price):
    tax = price * 0.1
    discount = price * 0.15
    return price + tax - discount

# Good - DRY implementation
from enum import Enum
from dataclasses import dataclass

class MembershipTier(Enum):
    BRONZE = 0.05
    SILVER = 0.10
    GOLD = 0.15
    PLATINUM = 0.20

@dataclass
class PriceCalculator:
    TAX_RATE: float = 0.10
    
    @staticmethod
    def calculate_final_price(price: float, tier: MembershipTier) -> float:
        """Single source of truth for price calculation"""
        tax = price * PriceCalculator.TAX_RATE
        discount = price * tier.value
        return price + tax - discount

# Usage
calculator = PriceCalculator()
print(calculator.calculate_final_price(100, MembershipTier.GOLD))
```

### Q2.3: Explain KISS (Keep It Simple, Stupid) principle with examples.

**Answer:**
KISS advocates for simplicity - the simplest solution is usually the best.

**Analogy:** Like using a hammer to hang a picture instead of building a complex pulley system.

**Example:**
```python
# Bad - Over-engineered
class AdvancedStringReverser:
    def __init__(self, string):
        self.string = string
        self.reversed_string = None
    
    def reverse(self):
        self.reversed_string = self._recursive_reverse(self.string)
        return self.reversed_string
    
    def _recursive_reverse(self, s):
        if len(s) <= 1:
            return s
        return self._recursive_reverse(s[1:]) + s[0]

reverser = AdvancedStringReverser("hello")
result = reverser.reverse()

# Good - Simple and clear
def reverse_string(s: str) -> str:
    """Simple string reversal"""
    return s[::-1]

result = reverse_string("hello")

# Another example: Finding max value
# Bad - Unnecessarily complex
def find_maximum_value(numbers):
    if not numbers:
        return None
    max_val = numbers[0]
    for i in range(1, len(numbers)):
        if numbers[i] > max_val:
            max_val = numbers[i]
    return max_val

# Good - Simple using built-in
def find_maximum(numbers):
    return max(numbers) if numbers else None
```

### Q2.4: How would you structure a machine learning project following clean code principles?

**Answer:**
```python
# Project Structure Example
"""
ml_project/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── validator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── classifier.py
│   │   └── regressor.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   └── config.py
│   └── api/
│       ├── __init__.py
│       └── inference.py
├── tests/
├── configs/
├── requirements.txt
└── README.md
"""

# data/loader.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Union
from pathlib import Path

class DataLoader(ABC):
    """Abstract base class for data loading - follows OCP"""
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load data from source"""
        pass

class CSVDataLoader(DataLoader):
    """Loads data from CSV files - SRP"""
    
    def __init__(self, delimiter: str = ','):
        self.delimiter = delimiter
    
    def load(self, path: Union[str, Path]) -> pd.DataFrame:
        return pd.read_csv(path, delimiter=self.delimiter)

class ParquetDataLoader(DataLoader):
    """Loads data from Parquet files"""
    
    def load(self, path: Union[str, Path]) -> pd.DataFrame:
        return pd.read_parquet(path)

# data/preprocessor.py
from sklearn.preprocessing import StandardScaler
import numpy as np

class DataPreprocessor:
    """Handles data preprocessing - SRP"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data"""
        self.is_fitted = True
        return self.scaler.fit_transform(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        return self.scaler.transform(X)

# models/base_model.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class BaseModel(ABC):
    """Abstract base for all models - follows ISP and DIP"""
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model"""
        pass

# training/trainer.py
from typing import Optional
import logging

class ModelTrainer:
    """Orchestrates model training - SRP"""
    
    def __init__(
        self,
        model: BaseModel,
        data_loader: DataLoader,
        preprocessor: DataPreprocessor,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.logger = logger or logging.getLogger(__name__)
    
    def train_pipeline(self, data_path: str) -> Dict[str, Any]:
        """Execute full training pipeline - KISS"""
        self.logger.info("Starting training pipeline")
        
        # Load data
        df = self.data_loader.load(data_path)
        
        # Prepare features
        X = df.drop('target', axis=1).values
        y = df['target'].values
        
        # Preprocess
        X_processed = self.preprocessor.fit_transform(X)
        
        # Train
        self.model.train(X_processed, y)
        
        self.logger.info("Training completed")
        return {"status": "success", "samples": len(X)}

# utils/config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration management - SRP"""
    model_type: str
    learning_rate: float
    batch_size: int
    epochs: int
    random_seed: int = 42
    checkpoint_dir: Optional[str] = None

# Usage example demonstrating clean code principles
def main():
    # Dependency injection - DIP
    data_loader = CSVDataLoader()
    preprocessor = DataPreprocessor()
    model = SomeModel()  # Implements BaseModel
    
    # Single responsibility classes working together
    trainer = ModelTrainer(
        model=model,
        data_loader=data_loader,
        preprocessor=preprocessor
    )
    
    # Simple, clear execution - KISS
    results = trainer.train_pipeline("data/train.csv")
    print(results)
```

---

## 3. Containerization (Docker)

### Q3.1: Explain Docker architecture and key components.

**Answer:**
Docker uses a client-server architecture with these key components:

**Components:**
- **Docker Client:** Interface for users (CLI)
- **Docker Daemon:** Builds, runs, and manages containers
- **Docker Images:** Read-only templates
- **Docker Containers:** Running instances of images
- **Docker Registry:** Stores images (Docker Hub)

**Analogy:** Think of Docker like a shipping container system:
- **Image** = Blueprint/template of the container
- **Container** = The actual shipping container with goods
- **Dockerfile** = Instructions for building the container
- **Registry** = Warehouse storing container blueprints

**Example:**
```dockerfile
# Dockerfile - No hardcoded secrets
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use environment variables for configuration
ENV MODEL_PATH=/app/models

# NEVER DO THIS - hardcoded secrets
# ENV API_KEY=sk-1234567890abcdef
# ENV DB_PASSWORD=mysecretpassword

# Run as non-root user
USER 1000

CMD ["python", "app.py"]
```

**Docker Compose with Secrets:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: .
    env_file:
      - .env  # Not committed to Git
    secrets:
      - db_password
      - api_key
    environment:
      - DB_PASSWORD_FILE=/run/secrets/db_password
      - API_KEY_FILE=/run/secrets/api_key

secrets:
  db_password:
    file: ./secrets/db_password.txt
  api_key:
    file: ./secrets/api_key.txt
```

**Python code to read secrets:**
```python
# app.py
import os
from pathlib import Path

def get_secret(secret_name: str, default: str = None) -> str:
    """Read secret from file or environment variable"""
    # Try reading from Docker secret file first
    secret_file = os.getenv(f"{secret_name.upper()}_FILE")
    if secret_file and Path(secret_file).exists():
        return Path(secret_file).read_text().strip()
    
    # Fall back to environment variable
    return os.getenv(secret_name.upper(), default)

# Usage
db_password = get_secret("db_password")
api_key = get_secret("api_key")

# .env file (add to .gitignore)
"""
DB_PASSWORD=your_password_here
API_KEY=your_api_key_here
MODEL_PATH=/app/models
REDIS_URL=redis://localhost:6379
"""
```

**Using External Secret Manager:**
```python
# Using AWS Secrets Manager
import boto3
import json
from functools import lru_cache

class SecretManager:
    def __init__(self, region_name: str = "us-east-1"):
        self.client = boto3.client('secretsmanager', region_name=region_name)
    
    @lru_cache(maxsize=128)
    def get_secret(self, secret_name: str) -> dict:
        """Retrieve secret from AWS Secrets Manager"""
        response = self.client.get_secret_value(SecretId=secret_name)
        return json.loads(response['SecretString'])

# Dockerfile
"""
FROM python:3.10-slim
# Install AWS CLI is not needed - use boto3
COPY requirements.txt .
RUN pip install boto3
"""
```

---

## 4. Deep Learning & SOTA Implementation

### Q4.1: How do you approach implementing a paper from scratch?

**Answer:**

**Systematic approach:**
1. **Read thoroughly:** Understand architecture, loss functions, training procedure
2. **Check for official code:** Look for author's implementation
3. **Start with basics:** Implement core components first
4. **Unit test:** Test each component independently
5. **Reproduce results:** Validate against paper's benchmarks

**Analogy:** Like building IKEA furniture - read instructions first, organize parts, build step by step, test stability at each stage.

**Example - Implementing Attention Mechanism:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention from "Attention Is All You Need"
    Paper: https://arxiv.org/abs/1706.03762
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, temperature: float, dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: torch.Tensor = None):
        """
        Args:
            q: Queries [batch, n_heads, seq_len, d_k]
            k: Keys [batch, n_heads, seq_len, d_k]
            v: Values [batch, n_heads, seq_len, d_v]
            mask: Optional mask [batch, 1, seq_len, seq_len]
        
        Returns:
            output: [batch, n_heads, seq_len, d_v]
            attn_weights: [batch, n_heads, seq_len, seq_len]
        """
        # Compute attention scores
        # QK^T / sqrt(d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        
        # Apply mask if provided (for padding or causal masking)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(
            temperature=math.sqrt(self.d_k),
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor = None):
        """
        Args:
            q, k, v: [batch, seq_len, d_model]
            mask: [batch, 1, seq_len, seq_len]
        """
        batch_size = q.size(0)
        residual = q
        
        # Linear projections and split into heads
        # [batch, seq_len, d_model] -> [batch, seq_len, n_heads, d_k]
        # -> [batch, n_heads, seq_len, d_k]
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        output, attn_weights = self.attention(q, k, v, mask)
        
        # Concatenate heads
        # [batch, n_heads, seq_len, d_k] -> [batch, seq_len, n_heads, d_k]
        # -> [batch, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(output)
        output = self.dropout(output)
        
        # Add residual and normalize
        output = self.layer_norm(output + residual)
        
        return output, attn_weights

# Unit tests
def test_attention():
    batch_size, seq_len, d_model, n_heads = 2, 10, 512, 8
    
    mha = MultiHeadAttention(d_model, n_heads)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, attn_weights = mha(x, x, x)
    
    # Assertions
    assert output.shape == (batch_size, seq_len, d_model)
    assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    print("✓ Attention tests passed")

test_attention()
```

### Q4.2: How do you optimize model inference for production?

**Answer:**

**Optimization techniques:**
1. **Quantization:** Reduce precision (FP32 → INT8)
2. **Pruning:** Remove unnecessary weights
3. **Knowledge Distillation:** Train smaller model
4. **Model Compilation:** ONNX, TensorRT
5. **Batching:** Process multiple requests together

**Example:**
```python
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, quantize_static, prepare_qat
import onnx
import onnxruntime as ort
import numpy as np
from typing import List, Tuple
import time

class OptimizedModelInference:
    """
    Comprehensive model optimization for production inference
    """
    
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...]):
        self.model = model
        self.input_shape = input_shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def benchmark(self, num_iterations: int = 100) -> dict:
        """Benchmark inference speed"""
        self.model.eval()
        dummy_input = torch.randn(1, *self.input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)
        
        elapsed = time.time() - start_time
        throughput = num_iterations / elapsed
        latency = elapsed / num_iterations * 1000  # ms
        
        return {
            'throughput': f"{throughput:.2f} samples/sec",
            'latency': f"{latency:.2f} ms",
            'device': str(self.device)
        }
    
    def apply_dynamic_quantization(self) -> nn.Module:
        """
        Dynamic quantization - converts weights to INT8
        Best for: LSTMs, Linear layers
        """
        quantized_model = quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM},
            dtype=torch.qint8
        )
        return quantized_model
    
    def export_to_onnx(self, output_path: str = "model.onnx"):
        """Export model to ONNX format for cross-platform inference"""
        self.model.eval()
        dummy_input = torch.randn(1, *self.input_shape).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model exported to {output_path}")
        return output_path
    
    def create_onnx_session(self, onnx_path: str) -> ort.InferenceSession:
        """Create optimized ONNX Runtime session"""
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = \
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Use GPU if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
            if torch.cuda.is_available() else ['CPUExecutionProvider']
        
        session = ort.InferenceSession(
            onnx_path,
            session_options,
            providers=providers
        )
        return session
    
    def batch_inference(self, inputs: List[np.ndarray], 
                       batch_size: int = 32) -> List[np.ndarray]:
        """Efficient batch inference"""
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_tensor = torch.tensor(np.stack(batch)).to(self.device)
            
            with torch.no_grad():
                output = self.model(batch_tensor)
            
            results.extend(output.cpu().numpy())
        
        return results

# Example: Complete optimization pipeline
class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden.squeeze(0))
        return output

# Optimization workflow
def optimize_model_for_production():
    # Original model
    model = TextClassifier(vocab_size=10000, embed_dim=128, num_classes=5)
    optimizer = OptimizedModelInference(model, input_shape=(50,))  # seq_len=50
    
    print("=== Original Model ===")
    print(optimizer.benchmark())
    
    # 1. Dynamic Quantization
    quantized_model = optimizer.apply_dynamic_quantization()
    print("\n=== Quantized Model ===")
    optimizer.model = quantized_model
    print(optimizer.benchmark())
    
    # 2. Export to ONNX
    onnx_path = optimizer.export_to_onnx("classifier.onnx")
    
    # 3. ONNX Runtime inference
    print("\n=== ONNX Runtime ===")
    session = optimizer.create_onnx_session(onnx_path)
    
    # Benchmark ONNX
    dummy_input = np.random.randint(0, 10000, (1, 50)).astype(np.int64)
    start = time.time()
    for _ in range(100):
        _ = session.run(None, {'input': dummy_input})
    elapsed = time.time() - start
    print(f"ONNX Latency: {elapsed/100*1000:.2f} ms")

# Knowledge Distillation Example
class KnowledgeDistillation:
    """
    Distill knowledge from large teacher model to small student model
    """
    
    def __init__(self, teacher: nn.Module, student: nn.Module, 
                 temperature: float = 3.0, alpha: float = 0.5):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Combined loss: soft targets from teacher + hard targets
        """
        # Soft targets (from teacher)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean')
        soft_loss *= (self.temperature ** 2)
        
        # Hard targets (ground truth)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
    
    def train_step(self, batch, optimizer):
        """Single training step"""
        inputs, labels = batch
        
        # Teacher predictions (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        
        # Student predictions
        student_logits = self.student(inputs)
        
        # Compute loss
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
```

### Q4.3: How do you handle large models that don't fit in GPU memory?

**Answer:**

**Strategies:**
1. **Gradient Accumulation:** Simulate larger batches
2. **Mixed Precision Training:** Use FP16
3. **Model Parallelism:** Split model across GPUs
4. **Gradient Checkpointing:** Trade compute for memory
5. **CPU Offloading:** Move some layers to CPU

**Analogy:** Like moving furniture - if it doesn't fit through the door, you can: split it into pieces (model parallelism), make multiple trips (gradient accumulation), or use a different path (CPU offloading).

**Example:**
```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import List

class MemoryEfficientTraining:
    """
    Techniques for training large models with limited GPU memory
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
    
    def gradient_accumulation_training(
        self, 
        dataloader, 
        optimizer,
        accumulation_steps: int = 4,
        epochs: int = 1
    ):
        """
        Gradient Accumulation: Simulate larger batch sizes
        
        If batch_size=8 and accumulation_steps=4:
        Effective batch_size = 8 * 4 = 32
        """
        self.model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            for i, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = nn.functional.cross_entropy(outputs, labels)
                
                # Normalize loss by accumulation steps
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights every accumulation_steps
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    print(f"Epoch {epoch}, Step {i+1}, Loss: {loss.item()}")
    
    def mixed_precision_training(self, dataloader, optimizer, epochs: int = 1):
        """
        Mixed Precision (FP16): Reduces memory by ~50%
        Uses float16 for forward/backward, float32 for weights
        """
        scaler = GradScaler()
        self.model.train()
        
        for epoch in range(epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Automatic mixed precision
                with autocast():
                    outputs = self.model(inputs)
                    loss = nn.functional.cross_entropy(outputs, labels)
                
                # Scaled backward pass
                scaler.scale(loss).backward()
                
                # Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step with scaling
                scaler.step(optimizer)
                scaler.update()
    
    def gradient_checkpointing(self):
        """
        Gradient Checkpointing: Trade compute for memory
        Recomputes activations during backward pass
        """
        from torch.utils.checkpoint import checkpoint
        
        class CheckpointedModel(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.layers = original_model.layers
            
            def forward(self, x):
                for layer in self.layers:
                    # Checkpoint each layer
                    x = checkpoint(layer, x)
                return x
        
        return CheckpointedModel(self.model)

# Model Parallelism Example
class ModelParallelResNet(nn.Module):
    """
    Split model across multiple GPUs
    """
    
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.device1 = torch.device('cuda:0')
        self.device2 = torch.device('cuda:1')
        
        # First half on GPU 0
        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        ).to(self.device1)
        
        # Middle layers on GPU 0
        self.seq2 = nn.Sequential(
            *[self._make_layer(64, 128, 2) for _ in range(3)]
        ).to(self.device1)
        
        # Second half on GPU 1
        self.seq3 = nn.Sequential(
            *[self._make_layer(128, 256, 2) for _ in range(4)]
        ).to(self.device2)
        
        # Final layers on GPU 1
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        ).to(self.device2)
    
    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Move through GPU 0
        x = self.seq1(x.to(self.device1))
        x = self.seq2(x)
        
        # Move to GPU 1
        x = self.seq3(x.to(self.device2))
        x = self.fc(x)
        
        return x

# DeepSpeed Integration for Very Large Models
def setup_deepspeed_config():
    """
    DeepSpeed configuration for training billion-parameter models
    """
    config = {
        "train_batch_size": 32,
        "gradient_accumulation_steps": 4,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16
        },
        "zero_optimization": {
            "stage": 2,  # Stage 2: Optimizer state partitioning
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "contiguous_gradients": True,
            "overlap_comm": True
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True
        }
    }
    return config

# Usage example
"""
import deepspeed

model = YourLargeModel()
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=setup_deepspeed_config()
)

for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
"""
```

### Q4.4: How do you implement custom loss functions and metrics?

**Answer:**

**Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Paper: https://arxiv.org/abs/1708.02002
    
    FL(p_t) = -α_t(1 - p_t)^γ log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, C] logits
            targets: [N] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    
    Dice = 2|X∩Y| / (|X| + |Y|)
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [N, C, H, W] predicted probabilities
            targets: [N, C, H, W] ground truth (one-hot encoded)
        """
        predictions = F.softmax(predictions, dim=1)
        
        # Flatten spatial dimensions
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / \
               (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for metric learning
    Used in Siamese networks
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, output1: torch.Tensor, output2: torch.Tensor, 
                label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output1, output2: Embeddings [N, D]
            label: [N] 1 if similar, 0 if dissimilar
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        loss = (label * torch.pow(euclidean_distance, 2) +
                (1 - label) * torch.pow(
                    torch.clamp(self.margin - euclidean_distance, min=0.0), 2
                ))
        
        return loss.mean()

# Custom Metrics
class MetricsCalculator:
    """Calculate various ML metrics"""
    
    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Classification accuracy"""
        pred_classes = predictions.argmax(dim=1)
        correct = (pred_classes == targets).sum().item()
        return correct / len(targets)
    
    @staticmethod
    def precision_recall_f1(predictions: torch.Tensor, targets: torch.Tensor,
                           num_classes: int):
        """
        Calculate precision, recall, F1 per class
        """
        pred_classes = predictions.argmax(dim=1)
        
        metrics = {}
        for class_id in range(num_classes):
            # True positives, false positives, false negatives
            tp = ((pred_classes == class_id) & (targets == class_id)).sum().item()
            fp = ((pred_classes == class_id) & (targets != class_id)).sum().item()
            fn = ((pred_classes != class_id) & (targets == class_id)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) \
                if (precision + recall) > 0 else 0
            
            metrics[f'class_{class_id}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return metrics
    
    @staticmethod
    def iou_score(predictions: torch.Tensor, targets: torch.Tensor, 
                  smooth: float = 1e-6) -> float:
        """
        Intersection over Union for segmentation
        """
        predictions = (predictions > 0.5).float()
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()

# Training loop with custom loss and metrics
def train_with_custom_loss():
    model = YourModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Use custom loss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    metrics = MetricsCalculator()
    
    for epoch in range(10):
        for batch_inputs, batch_targets in dataloader:
            # Forward pass
            outputs = model(batch_inputs)
            
            # Calculate loss
            loss = criterion(outputs, batch_targets)
            
            # Calculate metrics
            acc = metrics.accuracy(outputs, batch_targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")
```

---

## 5. Data Drift & Model Drift

### Q5.1: What is the difference between data drift and model drift?

**Answer:**

**Data Drift (Covariate Shift):**
- Input distribution changes over time: P(X) changes
- Example: E-commerce model trained on pre-pandemic data used during pandemic

**Concept Drift:**
- Relationship between inputs and outputs changes: P(Y|X) changes
- Example: Fraud patterns evolve, old fraud detection rules become obsolete

**Model Drift (Model Decay):**
- Model performance degrades over time
- Can be caused by data drift, concept drift, or both

**Analogy:** 
- **Data Drift:** Your recipe (model) stays the same, but ingredient quality changes (different flour brand)
- **Concept Drift:** Your recipe stays the same, but customer preferences change (people now prefer less sugar)
- **Model Drift:** Your cake (model output) quality decreases over time

**Visual Comparison:**
```
Time →
Data Drift:    Original Distribution → Shifted Distribution
               (Mean, variance change)

Concept Drift: X→Y relationship changes
               (Decision boundary shifts)

Model Drift:   Performance ↓↓↓ over time
               (Accuracy, F1 decrease)
```

**Example Implementation:**
```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class DriftDetector:
    """
    Comprehensive drift detection system
    """
    
    def __init__(self, reference_data: pd.DataFrame, 
                 target_col: str = None,
                 significance_level: float = 0.05):
        """
        Args:
            reference_data: Training/baseline data
            target_col: Target variable column name
            significance_level: Statistical significance threshold
        """
        self.reference_data = reference_data
        self.target_col = target_col
        self.significance_level = significance_level
        self.feature_cols = [col for col in reference_data.columns 
                           if col != target_col]
    
    def detect_data_drift_ks(self, current_data: pd.DataFrame) -> Dict:
        """
        Kolmogorov-Smirnov test for data drift detection
        Tests if two samples come from the same distribution
        
        Returns: Dictionary with drift detection results per feature
        """
        drift_results = {}
        
        for feature in self.feature_cols:
            if feature not in current_data.columns:
                continue
            
            # Get reference and current distributions
            ref_values = self.reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()
            
            # KS test
            ks_statistic, p_value = stats.ks_2samp(ref_values, curr_values)
            
            drift_results[feature] = {
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'drift_detected': p_value < self.significance_level,
                'severity': self._calculate_severity(ks_statistic)
            }
        
        return drift_results
    
    def detect_data_drift_psi(self, current_data: pd.DataFrame, 
                             n_bins: int = 10) -> Dict:
        """
        Population Stability Index (PSI) for drift detection
        
        PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        
        PSI < 0.1: No significant change
        0.1 ≤ PSI < 0.2: Moderate change
        PSI ≥ 0.2: Significant change
        """
        psi_results = {}
        
        for feature in self.feature_cols:
            if feature not in current_data.columns:
                continue
            
            # Create bins based on reference data
            ref_values = self.reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()
            
            # Define bins
            bins = np.linspace(ref_values.min(), ref_values.max(), n_bins + 1)
            bins[0] = -np.inf
            bins[-1] = np.inf
            
            # Calculate distributions
            ref_dist = np.histogram(ref_values, bins=bins)[0] / len(ref_values)
            curr_dist = np.histogram(curr_values, bins=bins)[0] / len(curr_values)
            
            # Add small constant to avoid division by zero
            ref_dist = ref_dist + 1e-10
            curr_dist = curr_dist + 1e-10
            
            # Calculate PSI
            psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
            
            psi_results[feature] = {
                'psi': psi,
                'status': self._psi_interpretation(psi)
            }
        
        return psi_results
    
    def detect_concept_drift(self, current_data: pd.DataFrame,
                           current_predictions: np.ndarray) -> Dict:
        """
        Detect concept drift by comparing prediction distributions
        and monitoring performance metrics
        """
        if self.target_col is None:
            return {"error": "No target column specified"}
        
        results = {}
        
        # Check if target is available in current data
        if self.target_col in current_data.columns:
            current_targets = current_data[self.target_col]
            
            # Compare target distributions
            ref_targets = self.reference_data[self.target_col]
            ks_stat, p_value = stats.ks_2samp(ref_targets, current_targets)
            
            results['target_distribution'] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'drift_detected': p_value < self.significance_level
            }
            
            # Calculate performance metrics
            from sklearn.metrics import accuracy_score, f1_score
            accuracy = accuracy_score(current_targets, current_predictions)
            f1 = f1_score(current_targets, current_predictions, average='weighted')
            
            results['performance'] = {
                'accuracy': accuracy,
                'f1_score': f1
            }
        
        return results
    
    def _calculate_severity(self, ks_statistic: float) -> str:
        """Map KS statistic to severity level"""
        if ks_statistic < 0.1:
            return "low"
        elif ks_statistic < 0.2:
            return "moderate"
        else:
            return "high"
    
    def _psi_interpretation(self, psi: float) -> str:
        """Interpret PSI value"""
        if psi < 0.1:
            return "no_significant_change"
        elif psi < 0.2:
            return "moderate_change"
        else:
            return "significant_change"
    
    def generate_drift_report(self, current_data: pd.DataFrame) -> str:
        """Generate comprehensive drift report"""
        ks_results = self.detect_data_drift_ks(current_data)
        psi_results = self.detect_data_drift_psi(current_data)
        
        report = "=== DRIFT DETECTION REPORT ===\n\n"
        report += f"Reference Data Size: {len(self.reference_data)}\n"
        report += f"Current Data Size: {len(current_data)}\n\n"
        
        # Data drift summary
        drifted_features = [f for f, r in ks_results.items() 
                          if r['drift_detected']]
        
        report += f"Data Drift Detected in {len(drifted_features)} features:\n"
        for feature in drifted_features:
            report += f"  - {feature}: KS={ks_results[feature]['ks_statistic']:.4f}, "
            report += f"PSI={psi_results[feature]['psi']:.4f}\n"
        
        return report

# Example: Monitoring system
class ModelMonitoringSystem:
    """
    Production monitoring system for model drift
    """
    
    def __init__(self, model, reference_data: pd.DataFrame, 
                 target_col: str, monitoring_window: int = 1000):
        self.model = model
        self.drift_detector = DriftDetector(reference_data, target_col)
        self.monitoring_window = monitoring_window
        self.performance_history = []
        self.drift_history = []
    
    def monitor_batch(self, new_data: pd.DataFrame, 
                     new_targets: np.ndarray = None) -> Dict:
        """Monitor a new batch of data"""
        # Get predictions
        predictions = self.model.predict(new_data.drop(self.drift_detector.target_col, 
                                                       axis=1, errors='ignore'))
        
        # Detect data drift
        data_drift = self.drift_detector.detect_data_drift_psi(new_data)
        
        # Calculate metrics
        results = {
            'timestamp': datetime.now(),
            'batch_size': len(new_data),
            'data_drift': data_drift,
            'predictions': predictions
        }
        
        # If we have ground truth, calculate performance
        if new_targets is not None:
            from sklearn.metrics import accuracy_score, f1_score
            accuracy = accuracy_score(new_targets, predictions)
            f1 = f1_score(new_targets, predictions, average='weighted')
            
            results['performance'] = {
                'accuracy': accuracy,
                'f1_score': f1
            }
            
            self.performance_history.append({
                'timestamp': datetime.now(),
                'accuracy': accuracy,
                'f1_score': f1
            })
        
        self.drift_history.append(results)
        
        # Check if retraining is needed
        if self._should_retrain(results):
            results['action'] = 'RETRAIN_RECOMMENDED'
        
        return results
    
    def _should_retrain(self, results: Dict) -> bool:
        """Determine if model retraining is needed"""
        # Check for significant data drift
        significant_drift_count = sum(
            1 for feature, result in results['data_drift'].items()
            if result['status'] == 'significant_change'
        )
        
        if significant_drift_count >= 3:
            return True
        
        # Check performance degradation
        if 'performance' in results and len(self.performance_history) >= 10:
            recent_perf = [h['accuracy'] for h in self.performance_history[-10:]]
            avg_recent = np.mean(recent_perf)
            baseline_perf = self.performance_history[0]['accuracy']
            
            if avg_recent < baseline_perf * 0.9:  # 10% drop
                return True
        
        return False
    
    def plot_performance_over_time(self):
        """Visualize model performance degradation"""
        if not self.performance_history:
            print("No performance history available")
            return
        
        timestamps = [h['timestamp'] for h in self.performance_history]
        accuracies = [h['accuracy'] for h in self.performance_history]
        f1_scores = [h['f1_score'] for h in self.performance_history]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(timestamps, accuracies, marker='o')
        plt.title('Accuracy Over Time')
        plt.xlabel('Time')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(timestamps, f1_scores, marker='o', color='orange')
        plt.title('F1 Score Over Time')
        plt.xlabel('Time')
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Practical Usage Example
def monitoring_workflow():
    """Complete drift monitoring workflow"""
    # 1. Train model with reference data
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate reference data
    X_ref, y_ref = make_classification(n_samples=10000, n_features=20, 
                                       n_informative=15, random_state=42)
    ref_df = pd.DataFrame(X_ref, columns=[f'feature_{i}' for i in range(20)])
    ref_df['target'] = y_ref
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(ref_df.drop('target', axis=1), ref_df['target'])
    
    # 2. Setup monitoring
    monitor = ModelMonitoringSystem(model, ref_df, 'target')
    
    # 3. Simulate drift over time
    for week in range(10):
        # Simulate data drift (shift mean)
        drift_factor = week * 0.1
        X_current, y_current = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            random_state=42 + week, shift=drift_factor
        )
        current_df = pd.DataFrame(X_current, 
                                 columns=[f'feature_{i}' for i in range(20)])
        current_df['target'] = y_current
        
        # Monitor
        results = monitor.monitor_batch(current_df, y_current)
        
        print(f"\n=== Week {week + 1} ===")
        print(f"Performance: {results.get('performance', {})}")
        print(f"Action: {results.get('action', 'CONTINUE_MONITORING')}")
    
    # 4. Visualize
    monitor.plot_performance_over_time()

# Usage
# monitoring_workflow()
```

### Q5.2: How do you set actionable metrics for drift detection?

**Answer:**

**Key Principles:**
1. **Align with business impact:** Metrics should tie to business KPIs
2. **Set appropriate thresholds:** Based on historical data and domain knowledge
3. **Consider multiple metrics:** Use ensemble of indicators
4. **Define clear actions:** Each threshold should trigger specific response

**Example:**
```python
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Callable
import numpy as np

class AlertLevel(Enum):
    GREEN = "no_action"
    YELLOW = "investigate"
    ORANGE = "prepare_retrain"
    RED = "immediate_retrain"

@dataclass
class DriftThreshold:
    """Define thresholds for drift metrics"""
    metric_name: str
    green_max: float
    yellow_max: float
    orange_max: float
    # Anything above orange_max is RED

@dataclass
class ActionableMetric:
    """Metric with associated thresholds and actions"""
    name: str
    calculate: Callable
    thresholds: DriftThreshold
    action_green: str
    action_yellow: str
    action_orange: str
    action_red: str

class ActionableDriftMonitoring:
    """
    Drift monitoring system with clear action triggers
    """
    
    def __init__(self):
        self.metrics = self._setup_metrics()
        self.alert_history = []
    
    def _setup_metrics(self) -> List[ActionableMetric]:
        """Configure all metrics with thresholds"""
        
        metrics = [
            # PSI for feature drift
            ActionableMetric(
                name="psi_critical_features",
                calculate=self._calculate_psi,
                thresholds=DriftThreshold(
                    metric_name="PSI",
                    green_max=0.10,   # No significant change
                    yellow_max=0.15,  # Minor drift
                    orange_max=0.25   # Moderate drift
                    # > 0.25 = Severe drift (RED)
                ),
                action_green="Continue normal monitoring",
                action_yellow="Increase monitoring frequency; Investigate features",
                action_orange="Prepare retraining pipeline; Collect fresh data",
                action_red="Trigger immediate retraining; Alert data team"
            ),
            
            # Model performance
            ActionableMetric(
                name="accuracy_drop",
                calculate=self._calculate_accuracy_drop,
                thresholds=DriftThreshold(
                    metric_name="Accuracy Drop %",
                    green_max=2.0,    # <2% drop acceptable
                    yellow_max=5.0,   # 2-5% drop: warning
                    orange_max=10.0   # 5-10% drop: serious
                    # >10% drop = critical (RED)
                ),
                action_green="Continue monitoring",
                action_yellow="Review recent predictions; Check data quality",
                action_orange="Schedule retraining within 48 hours",
                action_red="Rollback to previous model; Emergency retrain"
            ),
            
            # Prediction distribution
            ActionableMetric(
                name="prediction_distribution_shift",
                calculate=self._calculate_pred_dist_shift,
                thresholds=DriftThreshold(
                    metric_name="KS Statistic",
                    green_max=0.05,
                    yellow_max=0.10,
                    orange_max=0.20
                ),
                action_green="No action needed",
                action_yellow="Analyze prediction patterns",
                action_orange="Review model assumptions; Check for concept drift",
                action_red="Immediate investigation; Consider model replacement"
            ),
            
            # Feature importance drift
            ActionableMetric(
                name="feature_importance_change",
                calculate=self._calculate_feature_importance_drift,
                thresholds=DriftThreshold(
                    metric_name="Importance Change",
                    green_max=0.10,
                    yellow_max=0.20,
                    orange_max=0.35
                ),
                action_green="Continue monitoring",
                action_yellow="Document changes; Update feature monitoring",
                action_orange="Review feature engineering; Check data sources",
                action_red="Full model audit; Retrain with new features"
            )
        ]
        
        return metrics
    
    def evaluate_drift(self, reference_data: pd.DataFrame,
                      current_data: pd.DataFrame,
                      model) -> Dict:
        """
        Evaluate all metrics and determine actions
        """
        results = {
            'timestamp': datetime.now(),
            'metrics': {},
            'overall_alert_level': AlertLevel.GREEN,
            'recommended_actions': []
        }
        
        highest_alert = AlertLevel.GREEN
        
        for metric in self.metrics:
            # Calculate metric value
            value = metric.calculate(reference_data, current_data, model)
            
            # Determine alert level
            alert_level = self._get_alert_level(value, metric.thresholds)
            
            # Get recommended action
            action = self._get_action(alert_level, metric)
            
            results['metrics'][metric.name] = {
                'value': value,
                'alert_level': alert_level,
                'action': action
            }
            
            # Update overall alert level
            if alert_level.value > highest_alert.value:
                highest_alert = alert_level
            
            # Add to recommended actions
            if alert_level != AlertLevel.GREEN:
                results['recommended_actions'].append(action)
        
        results['overall_alert_level'] = highest_alert
        
        # Log alert
        self.alert_history.append(results)
        
        return results
    
    def _get_alert_level(self, value: float, 
                        thresholds: DriftThreshold) -> AlertLevel:
        """Determine alert level based on value and thresholds"""
        if value <= thresholds.green_max:
            return AlertLevel.GREEN
        elif value <= thresholds.yellow_max:
            return AlertLevel.YELLOW
        elif value <= thresholds.orange_max:
            return AlertLevel.ORANGE
        else:
            return AlertLevel.RED
    
    def _get_action(self, alert_level: AlertLevel, 
                   metric: ActionableMetric) -> str:
        """Get recommended action for alert level"""
        action_map = {
            AlertLevel.GREEN: metric.action_green,
            AlertLevel.YELLOW: metric.action_yellow,
            AlertLevel.ORANGE: metric.action_orange,
            AlertLevel.RED: metric.action_red
        }
        return action_map[alert_level]
    
    # Metric calculation methods
    def _calculate_psi(self, ref_data, curr_data, model) -> float:
        """Calculate PSI for critical features"""
        # Implementation from previous example
        return 0.12  # Placeholder
    
    def _calculate_accuracy_drop(self, ref_data, curr_data, model) -> float:
        """Calculate percentage drop in accuracy"""
        # Baseline accuracy
        baseline_acc = 0.95
        
        # Current accuracy (if labels available)
        if 'target' in curr_data.columns:
            from sklearn.metrics import accuracy_score
            X_curr = curr_data.drop('target', axis=1)
            y_curr = curr_data['target']
            preds = model.predict(X_curr)
            current_acc = accuracy_score(y_curr, preds)
            
            drop_percentage = ((baseline_acc - current_acc) / baseline_acc) * 100
            return drop_percentage
        
        return 0.0
    
    def _calculate_pred_dist_shift(self, ref_data, curr_data, model) -> float:
        """Calculate prediction distribution shift"""
        from scipy.stats import ks_2samp
        
        X_ref = ref_data.drop('target', axis=1, errors='ignore')
        X_curr = curr_data.drop('target', axis=1, errors='ignore')
        
        pred_ref = model.predict_proba(X_ref)[:, 1]
        pred_curr = model.predict_proba(X_curr)[:, 1]
        
        ks_stat, _ = ks_2samp(pred_ref, pred_curr)
        return ks_stat
    
    def _calculate_feature_importance_drift(self, ref_data, curr_data, 
                                           model) -> float:
        """Calculate change in feature importance"""
        # Get current feature importance
        if hasattr(model, 'feature_importances_'):
            current_importance = model.feature_importances_
            
            # Compare with baseline (stored separately)
            baseline_importance = np.array([0.1] * len(current_importance))
            
            # Calculate total variation distance
            change = np.sum(np.abs(current_importance - baseline_importance))
            return change
        
        return 0.0
    
    def generate_action_report(self, evaluation_results: Dict) -> str:
        """Generate human-readable action report"""
        report = "="* 60 + "\n"
        report += "DRIFT MONITORING ALERT REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Timestamp: {evaluation_results['timestamp']}\n"
        report += f"Overall Alert Level: {evaluation_results['overall_alert_level'].name}\n\n"
        
        report += "METRICS SUMMARY:\n"
        report += "-" * 60 + "\n"
        
        for metric_name, metric_data in evaluation_results['metrics'].items():
            report += f"\n{metric_name}:\n"
            report += f"  Value: {metric_data['value']:.4f}\n"
            report += f"  Alert: {metric_data['alert_level'].name}\n"
            report += f"  Action: {metric_data['action']}\n"
        
        report += "\n" + "=" * 60 + "\n"
        report += "RECOMMENDED ACTIONS:\n"
        report += "=" * 60 + "\n"
        
        for i, action in enumerate(evaluation_results['recommended_actions'], 1):
            report += f"{i}. {action}\n"
        
        return report

# Integration with alerting system
class AlertingSystem:
    """Send alerts based on drift detection"""
    
    def __init__(self):
        self.channels = {
            AlertLevel.YELLOW: ['slack'],
            AlertLevel.ORANGE: ['slack', 'email'],
            AlertLevel.RED: ['slack', 'email', 'pagerduty']
        }
    
    def send_alert(self, alert_level: AlertLevel, message: str):
        """Send alert through appropriate channels"""
        if alert_level == AlertLevel.GREEN:
            return  # No alert needed
        
        channels = self.channels.get(alert_level, [])
        
        for channel in channels:
            if channel == 'slack':
                self._send_slack(alert_level, message)
            elif channel == 'email':
                self._send_email(alert_level, message)
            elif channel == 'pagerduty':
                self._send_pagerduty(alert_level, message)
    
    def _send_slack(self, alert_level: AlertLevel, message: str):
        """Send Slack notification"""
        # Implementation
        print(f"[SLACK] {alert_level.name}: {message}")
    
    def _send_email(self, alert_level: AlertLevel, message: str):
        """Send email notification"""
        # Implementation
        print(f"[EMAIL] {alert_level.name}: {message}")
    
    def _send_pagerduty(self, alert_level: AlertLevel, message: str):
        """Trigger PagerDuty incident"""
        # Implementation
        print(f"[PAGERDUTY] {alert_level.name}: {message}")

# Complete workflow
def production_monitoring_workflow():
    """End-to-end monitoring with actionable metrics"""
    
    # Setup
    monitor = ActionableDriftMonitoring()
    alerting = AlertingSystem()
    
    # Evaluate drift
    evaluation = monitor.evaluate_drift(ref_data, current_data, model)
    
    # Generate report
    report = monitor.generate_action_report(evaluation)
    print(report)
    
    # Send alerts if needed
    if evaluation['overall_alert_level'] != AlertLevel.GREEN:
        alerting.send_alert(
            evaluation['overall_alert_level'],
            report
        )
    
    # Take automated actions
    if evaluation['overall_alert_level'] == AlertLevel.RED:
        # Trigger retraining pipeline
        trigger_retraining_pipeline()
    elif evaluation['overall_alert_level'] == AlertLevel.ORANGE:
        # Schedule retraining
        schedule_retraining(within_hours=48)
```

---

## 6. Application Architecture & Integration

### Q6.1: Explain microservices architecture for ML systems.

**Answer:**

**Microservices** = Decomposing application into small, independent services that communicate via APIs.

**Benefits:**
- Independent scaling
- Technology flexibility
- Fault isolation
- Easier deployment

**Analogy:** Like a restaurant with specialized stations (appetizers, main course, desserts) vs. one chef doing everything.

**ML Microservices Pattern:**
```
Client Request
     ↓
API Gateway
     ↓
├─→ Feature Service (Data preprocessing)
├─→ Model Service (Inference)
├─→ Monitoring Service (Logging, metrics)
└─→ Feedback Service (Model improvement)
```

**Example Implementation:**
```python
# architecture.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import httpx
from typing import List, Dict, Any
import redis
import pickle
import numpy as np

# ==================== Feature Service ====================
# Microservice 1: Feature Engineering
app_features = FastAPI(title="Feature Service")

class FeatureRequest(BaseModel):
    user_id: str
    raw_data: Dict[str, Any]

class FeatureResponse(BaseModel):
    features: List[float]
    feature_names: List[str]

@app_features.post("/extract-features")
async def extract_features(request: FeatureRequest) -> FeatureResponse:
    """Extract and engineer features from raw data"""
    # Feature engineering logic
    features = [
        request.raw_data.get('age', 0) / 100.0,
        request.raw_data.get('income', 0) / 100000.0,
        len(request.raw_data.get('purchase_history', [])),
        # ... more features
    ]
    
    feature_names = ['age_normalized', 'income_normalized', 'purchase_count']
    
    return FeatureResponse(features=features, feature_names=feature_names)

# ==================== Model Service ====================
# Microservice 2: Model Inference
app_model = FastAPI(title="Model Inference Service")

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str

class ModelService:
    def __init__(self):
        self.model = None
        self.model_version = "v1.0"
        self.cache = redis.Redis(host='redis', port=6379, decode_responses=True)
    
    def load_model(self):
        """Load model from storage"""
        with open('model.pkl', 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, features: List[float]) -> Dict:
        """Make prediction"""
        # Check cache first
        cache_key = f"pred:{hash(tuple(features))}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            return eval(cached_result)
        
        # Make prediction
        X = np.array(features).reshape(1, -1)
        prediction = self.model.predict(X)[0]
        confidence = self.model.predict_proba(X).max()
        
        result = {
            'prediction': float(prediction),
            'confidence': float(confidence),
            'model_version': self.model_version
        }
        
        # Cache result
        self.cache.setex(cache_key, 3600, str(result))
        
        return result

model_service = ModelService()

@app_model.on_event("startup")
async def startup():
    model_service.load_model()

@app_model.post("/predict")
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Get model prediction"""
    result = model_service.predict(request.features)
    return PredictionResponse(**result)

@app_model.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_service.model is not None}

# ==================== Monitoring Service ====================
# Microservice 3: Logging and Monitoring
app_monitoring = FastAPI(title="Monitoring Service")

class PredictionLog(BaseModel):
    user_id: str
    features: List[float]
    prediction: float
    confidence: float
    timestamp: str

class MonitoringService:
    def __init__(self):
        self.db = []  # In production: use proper database
        self.metrics = {
            'total_predictions': 0,
            'avg_confidence': 0.0
        }
    
    def log_prediction(self, log: PredictionLog):
        """Log prediction for monitoring"""
        self.db.append(log.dict())
        self.metrics['total_predictions'] += 1
        
        # Update running average
        n = self.metrics['total_predictions']
        current_avg = self.metrics['avg_confidence']
        self.metrics['avg_confidence'] = (
            (current_avg * (n - 1) + log.confidence) / n
        )
    
    def get_metrics(self) -> Dict:
        """Get monitoring metrics"""
        return self.metrics

monitoring_service = MonitoringService()

@app_monitoring.post("/log")
async def log_prediction(log: PredictionLog):
    """Log prediction"""
    monitoring_service.log_prediction(log)
    return {"status": "logged"}

@app_monitoring.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return monitoring_service.get_metrics()

# ==================== API Gateway ====================
# Orchestrates all microservices
app_gateway = FastAPI(title="ML API Gateway")

class MLRequest(BaseModel):
    user_id: str
    raw_data: Dict[str, Any]

class MLResponse(BaseModel):
    user_id: str
    prediction: float
    confidence: float
    model_version: str

FEATURE_SERVICE_URL = "http://feature-service:8001"
MODEL_SERVICE_URL = "http://model-service:8002"
MONITORING_SERVICE_URL = "http://monitoring-service:8003"

@app_gateway.post("/predict", response_model=MLResponse)
async def predict_endpoint(request: MLRequest):
    """
    Main prediction endpoint - orchestrates all services
    """
    async with httpx.AsyncClient() as client:
        try:
            # Step 1: Extract features
            feature_response = await client.post(
                f"{FEATURE_SERVICE_URL}/extract-features",
                json=request.dict(),
                timeout=5.0
            )
            feature_response.raise_for_status()
            features = feature_response.json()
            
            # Step 2: Get prediction
            prediction_response = await client.post(
                f"{MODEL_SERVICE_URL}/predict",
                json={"features": features['features']},
                timeout=5.0
            )
            prediction_response.raise_for_status()
            prediction = prediction_response.json()
            
            # Step 3: Log for monitoring (fire and forget)
            from datetime import datetime
            await client.post(
                f"{MONITORING_SERVICE_URL}/log",
                json={
                    "user_id": request.user_id,
                    "features": features['features'],
                    "prediction": prediction['prediction'],
                    "confidence": prediction['confidence'],
                    "timestamp": datetime.now().isoformat()
                },
                timeout=2.0
            )
            
            return MLResponse(
                user_id=request.user_id,
                prediction=prediction['prediction'],
                confidence=prediction['confidence'],
                model_version=prediction['model_version']
            )
            
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Service timeout")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Service error: {str(e)}")

@app_gateway.get("/health")
async def gateway_health():
    """Check health of all services"""
    async with httpx.AsyncClient() as client:
        services_health = {}
        
        services = {
            "feature": FEATURE_SERVICE_URL,
            "model": MODEL_SERVICE_URL,
            "monitoring": MONITORING_SERVICE_URL
        }
        
        for name, url in services.items():
            try:
                response = await client.get(f"{url}/health", timeout=2.0)
                services_health[name] = response.json()
            except:
                services_health[name] = {"status": "unhealthy"}
        
        all_healthy = all(s.get("status") == "healthy" 
                         for s in services_health.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "services": services_health
        }

# Docker Compose for entire system
"""
# docker-compose.yml
version: '3.8'

services:
  api-gateway:
    build: ./gateway
    ports:
      - "8000:8000"
    environment:
      - FEATURE_SERVICE_URL=http://feature-service:8001
      - MODEL_SERVICE_URL=http://model-service:8002
      - MONITORING_SERVICE_URL=http://monitoring-service:8003
    depends_on:
      - feature-service
      - model-service
      - monitoring-service

  feature-service:
    build: ./feature-service
    ports:
      - "8001:8001"

  model-service:
    build: ./model-service
    ports:
      - "8002:8002"
    volumes:
      - ./models:/models
    depends_on:
      - redis

  monitoring-service:
    build: ./monitoring-service
    ports:
      - "8003:8003"
    depends_on:
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_DB: monitoring
      POSTGRES_USER: mluser
      POSTGRES_PASSWORD: mlpassword
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
"""
```

### Q6.2: What's the difference between batch and real-time processing?

**Answer:**

**Batch Processing:**
- Processes data in large chunks at scheduled intervals
- Higher latency (minutes to hours)
- Better for high throughput
- Example: Daily recommendation updates, monthly reports

**Real-time (Streaming) Processing:**
- Processes data as it arrives
- Low latency (milliseconds to seconds)
- Better for immediate responses
- Example: Fraud detection, real-time recommendations

**Analogy:**
- **Batch:** Like doing laundry once a week (efficient but delayed)
- **Real-time:** Like washing dishes immediately after use (immediate but more effort)

**Example:**
```python
# ==================== Batch Processing ====================
import pandas as pd
from datetime import datetime, timedelta
import schedule
import time

class BatchMLPipeline:
    """
    Batch processing pipeline for ML predictions
    """
    
    def __init__(self, model, batch_size: int = 1000):
        self.model = model
        self.batch_size = batch_size
        self.results_store = []
    
    def load_batch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load data for batch processing"""
        # Simulate loading from database
        query = f"""
        SELECT * FROM user_events
        WHERE event_timestamp BETWEEN '{start_date}' AND '{end_date}'
        """
        # df = pd.read_sql(query, connection)
        df = pd.DataFrame()  # Placeholder
        return df
    
    def preprocess_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess entire batch"""
        # Feature engineering
        df['feature_1'] = df['raw_value'] / df['raw_value'].max()
        df['feature_2'] = df.groupby('user_id')['value'].transform('mean')
        return df
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for entire batch"""
        features = df[['feature_1', 'feature_2']].values
        predictions = self.model.predict(features)
        df['prediction'] = predictions
        return df
    
    def save_results(self, df: pd.DataFrame):
        """Save batch results to database"""
        # df.to_sql('predictions', connection, if_exists='append')
        self.results_store.append(df)
        print(f"Saved {len(df)} predictions")
    
    def run_batch_job(self):
        """Execute complete batch pipeline"""
        print(f"Starting batch job at {datetime.now()}")
        
        # Process yesterday's data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        # Load data
        df = self.load_batch_data(start_date, end_date)
        print(f"Loaded {len(df)} records")
        
        # Process in chunks
        for i in range(0, len(df), self.batch_size):
            chunk = df.iloc[i:i + self.batch_size]
            
            # Preprocess
            chunk = self.preprocess_batch(chunk)
            
            # Predict
            chunk = self.predict_batch(chunk)
            
            # Save
            self.save_results(chunk)
        
        print(f"Batch job completed at {datetime.now()}")
    
    def schedule_batch_jobs(self):
        """Schedule batch jobs"""
        # Run daily at 2 AM
        schedule.every().day.at("02:00").do(self.run_batch_job)
        
        print("Batch job scheduler started")
        while True:
            schedule.run_pending()
            time.sleep(60)

# ==================== Real-time Processing ====================
import asyncio
from kafka import KafkaConsumer, KafkaProducer
import json

class RealtimeMLPipeline:
    """
    Real-time streaming pipeline for ML predictions
    """
    
    def __init__(self, model):
        self.model = model
        self.consumer = KafkaConsumer(
            'input-events',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest'
        )
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda m: json.dumps(m).encode('utf-8')
        )
        self.feature_cache = {}
    
    def extract_features(self, event: Dict) -> List[float]:
        """Extract features from single event"""
        user_id = event['user_id']
        
        # Get cached features
        cached_features = self.feature_cache.get(user_id, {})
        
        # Compute real-time features
        features = [
            event['value'] / 100.0,
            cached_features.get('historical_avg', 0),
            len(event.get('tags', []))
        ]
        
        return features
    
    def predict_single(self, features: List[float]) -> Dict:
        """Make prediction for single event"""
        X = np.array(features).reshape(1, -1)
        prediction = self.model.predict(X)[0]
        confidence = self.model.predict_proba(X).max()
        
        return {
            'prediction': float(prediction),
            'confidence': float(confidence)
        }
    
    async def process_event(self, event: Dict):
        """Process single event in real-time"""
        start_time = time.time()
        
        try:
            # Extract features
            features = self.extract_features(event)
            
            # Make prediction
            result = self.predict_single(features)
            
            # Add metadata
            result['user_id'] = event['user_id']
            result['event_id'] = event['event_id']
            result['latency_ms'] = (time.time() - start_time) * 1000
            
            # Send to output topic
            self.producer.send('predictions', value=result)
            
            print(f"Processed event {event['event_id']} in {result['latency_ms']:.2f}ms")
            
        except Exception as e:
            print(f"Error processing event: {e}")
    
    async def run_streaming(self):
        """Run real-time streaming pipeline"""
        print("Starting real-time pipeline")
        
        for message in self.consumer:
            event = message.value
            await self.process_event(event)

# ==================== Lambda Architecture ====================
class LambdaArchitecture:
    """
    Combines batch and streaming for best of both worlds
    
    - Batch Layer: Precomputes views from historical data
    - Speed Layer: Processes recent data in real-time
    - Serving Layer: Merges batch and speed layer results
    """
    
    def __init__(self, batch_pipeline, realtime_pipeline):
        self.batch_pipeline = batch_pipeline
        self.realtime_pipeline = realtime_pipeline
        self.batch_results = {}  # Precomputed batch results
        self.realtime_cache = {}  # Recent real-time results
    
    def get_prediction(self, user_id: str, event_data: Dict) -> Dict:
        """
        Get prediction by merging batch and real-time layers
        """
        # Get batch layer result (precomputed features)
        batch_features = self.batch_results.get(user_id, {})
        
        # Get speed layer result (recent events)
        realtime_features = self.realtime_cache.get(user_id, {})
        
        # Merge features
        combined_features = {
            **batch_features,
            **realtime_features,
            'current_event': event_data
        }
        
        # Make prediction with combined features
        features = self._prepare_features(combined_features)
        prediction = self.realtime_pipeline.predict_single(features)
        
        return prediction
    
    def _prepare_features(self, combined_features: Dict) -> List[float]:
        """Prepare features from combined batch and real-time data"""
        return [
            combined_features.get('batch_avg', 0),
            combined_features.get('realtime_count', 0),
            combined_features['current_event'].get('value', 0)
        ]

# ==================== Comparison ====================
def compare_batch_vs_realtime():
    """
    When to use batch vs real-time:
    
    BATCH PROCESSING - Use when:
    - High throughput more important than latency
    - Data arrives in large volumes periodically
    - Complex aggregations needed
    - Cost optimization important (process during off-peak)
    - Examples: Recommendation systems, report generation, model training
    
    REAL-TIME PROCESSING - Use when:
    - Low latency critical (<100ms)
    - Immediate action required
    - Event-driven architecture
    - Examples: Fraud detection, dynamic pricing, chatbots
    
    LAMBDA (BOTH) - Use when:
    - Need both historical context and real-time data
    - Want to balance accuracy and latency
    - Complex feature engineering
    - Examples: Personalized feeds, advanced recommenders
    """
    pass

# Example: Fraud Detection (Real-time)
class FraudDetectionRT:
    def __init__(self, model):
        self.model = model
        self.recent_transactions = {}  # user_id -> last 10 transactions
    
    async def check_transaction(self, transaction: Dict) -> Dict:
        """Real-time fraud check"""
        user_id = transaction['user_id']
        
        # Get user's recent transaction history
        history = self.recent_transactions.get(user_id, [])
        
        # Extract real-time features
        features = [
            transaction['amount'],
            transaction['amount'] / (np.mean([t['amount'] for t in history]) + 1),
            len(history),
            transaction['is_international'],
            transaction['hour_of_day']
        ]
        
        # Predict fraud probability
        fraud_prob = self.model.predict_proba([features])[0][1]
        
        # Update history
        history.append(transaction)
        self.recent_transactions[user_id] = history[-10:]  # Keep last 10
        
        # Immediate decision
        is_fraud = fraud_prob > 0.7
        action = "BLOCK" if is_fraud else "ALLOW"
        
        return {
            'transaction_id': transaction['id'],
            'fraud_probability': fraud_prob,
            'action': action,
            'processing_time_ms': 15  # Must be < 50ms for real-time
        }

# Example: Recommendation System (Batch)
class RecommendationSystemBatch:
    def __init__(self, model):
        self.model = model
    
    def generate_daily_recommendations(self, all_users: List[str]):
        """Batch generate recommendations for all users"""
        recommendations = {}
        
        # Process in large batches (can take hours)
        for user_id in all_users:
            # Load user history (expensive query)
            user_history = self.load_user_history(user_id)
            
            # Generate features (complex aggregations)
            features = self.compute_complex_features(user_history)
            
            # Get recommendations
            recs = self.model.predict(features)
            
            # Store for serving
            recommendations[user_id] = recs
        
        # Save to cache (Redis) for fast serving
        self.save_to_cache(recommendations)
        
        return recommendations
    
    def load_user_history(self, user_id: str) -> pd.DataFrame:
        """Load complete user history"""
        # Can take seconds - OK for batch
        return pd.DataFrame()
    
    def compute_complex_features(self, history: pd.DataFrame) -> List[float]:
        """Complex feature engineering"""
        # Can use expensive operations - OK for batch
        return []
    
    def save_to_cache(self, recommendations: Dict):
        """Save precomputed recommendations"""
        # Serve these throughout the day
        pass
```

### Q6.3: How do you design a scalable ML API?

**Answer:**

**Key principles:**
1. **Stateless services:** No session data stored
2. **Caching:** Cache frequent predictions
3. **Load balancing:** Distribute requests
4. **Async processing:** Non-blocking operations
5. **Rate limiting:** Prevent abuse
6. **Monitoring:** Track performance

**Example:**
```python
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import time
import hashlib
import pickle
from prometheus_client import Counter, Histogram, generate_latest
import logging

# ==================== Scalable ML API ====================
app = FastAPI(title="Scalable ML API")

# Metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class PredictionRequest(BaseModel):
    features: List[float]
    user_id: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    cached: bool = False
    latency_ms: float

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]

# Cache configuration
class CacheManager:
    """Manages prediction caching with Redis"""
    
    def __init__(self):
        self.redis_client = None
        self.cache_ttl = 3600  # 1 hour
    
    async def init(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(
            "redis://localhost:6379",
            encoding="utf-8",
            decode_responses=False
        )
    
    def _get_cache_key(self, features: List[float]) -> str:
        """Generate cache key from features"""
        features_str = str(sorted(features))
        return f"pred:{hashlib.md5(features_str.encode()).hexdigest()}"
    
    async def get_cached_prediction(self, features: List[float]) -> Optional[Dict]:
        """Get cached prediction"""
        key = self._get_cache_key(features)
        cached = await self.redis_client.get(key)
        
        if cached:
            CACHE_HITS.inc()
            return pickle.loads(cached)
        return None
    
    async def cache_prediction(self, features: List[float], result: Dict):
        """Cache prediction result"""
        key = self._get_cache_key(features)
        await self.redis_client.setex(
            key,
            self.cache_ttl,
            pickle.dumps(result)
        )

# Model Manager (singleton)
class ModelManager:
    """Manages model lifecycle"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance.model_version = "v1.0"
        return cls._instance
    
    async def load_model(self):
        """Load model asynchronously"""
        # Simulate model loading
        await asyncio.sleep(0.1)
        # self.model = pickle.load(open('model.pkl', 'rb'))
        logger.info(f"Model {self.model_version} loaded")
    
    async def predict(self, features: List[float]) -> Dict:
        """Make prediction"""
        # Simulate inference
        await asyncio.sleep(0.01)  # 10ms inference time
        
        prediction = float(np.random.random())
        confidence = float(np.random.random())
        
        return {
            'prediction': prediction,
            'confidence': confidence
        }

# Initialize services
cache_manager = CacheManager()
model_manager = ModelManager()

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await cache_manager.init()
    await model_manager.load_model()
    await FastAPILimiter.init(await redis.from_url("redis://localhost:6379"))
    logger.info("API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await cache_manager.redis_client.close()
    logger.info("API shutdown complete")

# Dependency for rate limiting
async def get_user_id(request: Request) -> str:
    """Extract user ID from request"""
    return request.headers.get("X-User-ID", "anonymous")

# Endpoints
@app.post("/predict", response_model=PredictionResponse)
@PREDICTION_LATENCY.time()
async def predict(
    request: PredictionRequest,
    user_id: str = Depends(get_user_id),
    rate_limit: RateLimiter = Depends(RateLimiter(times=100, seconds=60))
):
    """
    Make single prediction
    Rate limited to 100 requests per minute per user
    """
    start_time = time.time()
    PREDICTION_COUNTER.inc()
    
    try:
        # Check cache first
        cached_result = await cache_manager.get_cached_prediction(request.features)
        
        if cached_result:
            cached_result['cached'] = True
            cached_result['latency_ms'] = (time.time() - start_time) * 1000
            return PredictionResponse(**cached_result)
        
        # Make prediction
        result = await model_manager.predict(request.features)
        
        # Cache result
        await cache_manager.cache_prediction(request.features, result)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            **result,
            cached=False,
            latency_ms=latency_ms
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/batch-predict")
async def batch_predict(
    batch_request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Batch prediction endpoint
    Process multiple requests efficiently
    """
    # Process requests concurrently
    tasks = [
        model_manager.predict(req.features)
        for req in batch_request.requests
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Log batch completion in background
    background_tasks.add_task(
        logger.info,
        f"Batch prediction completed: {len(results)} predictions"
    )
    
    return {"predictions": results, "count": len(results)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_version": model_manager.model_version,
        "cache_connected": cache_manager.redis_client is not None
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

# Load balancing configuration
"""
# nginx.conf
upstream ml_api {
    least_conn;  # Use least connections algorithm
    server ml-api-1:8000 weight=1 max_fails=3 fail_timeout=30s;
    server ml-api-2:8000 weight=1 max_fails=3 fail_timeout=30s;
    server ml-api-3:8000 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://ml_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
    }
}
"""

# Docker Compose with scaling
"""
# docker-compose.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - ml-api

  ml-api:
    build: .
    deploy:
      replicas: 3  # Scale to 3 instances
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 1gb --maxmemory-policy allkeys-lru

# Scale command:
# docker-compose up --scale ml-api=5
"""

# Circuit Breaker Pattern
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """
    Circuit breaker pattern for resilient API calls
    """
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Service temporarily unavailable"
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Reset on successful call"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

---

## 7. Orchestration Frameworks

### Q7.1: Explain LangChain and its core concepts.

**Answer:**

**LangChain** is a framework for building applications with Large Language Models (LLMs).

**Core Concepts:**
1. **Chains:** Sequence of calls to LLMs or other utilities
2. **Agents:** LLMs that make decisions about which actions to take
3. **Memory:** Persist state between chain/agent calls
4. **Retrievers:** Interface for retrieving relevant documents
5. **Tools:** Functions that agents can use

**Analogy:** LangChain is like a workflow automation tool for AI - it connects different AI components (LLMs, databases, APIs) into a cohesive application.

**Example:**
```python
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import numpy as np

# ==================== Basic Chain ====================
class SimpleLangChainExample:
    """Basic LangChain usage"""
    
    def __init__(self, api_key: str):
        self.llm = OpenAI(temperature=0.7, openai_api_key=api_key)
    
    def simple_chain(self):
        """Simple prompt-based chain"""
        template = """
        You are a helpful ML engineer assistant.
        
        Question: {question}
        
        Answer: Let me help you with that.
        """
        
        prompt = PromptTemplate(
            input_variables=["question"],
            template=template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        response = chain.run(question="How do I prevent overfitting?")
        return response
    
    def sequential_chain(self):
        """Chain multiple steps together"""
        from langchain.chains import SimpleSequentialChain
        
        # Step 1: Generate code
        code_template = """
        Write Python code to: {task}
        
        Code:
        """
        code_prompt = PromptTemplate(
            input_variables=["task"],
            template=code_template
        )
        code_chain = LLMChain(llm=self.llm, prompt=code_prompt)
        
        # Step 2: Explain code
        explain_template = """
        Explain this code:
        
        {code}
        
        Explanation:
        """
        explain_prompt = PromptTemplate(
            input_variables=["code"],
            template=explain_template
        )
        explain_chain = LLMChain(llm=self.llm, prompt=explain_prompt)
        
        # Combine chains
        overall_chain = SimpleSequentialChain(
            chains=[code_chain, explain_chain],
            verbose=True
        )
        
        result = overall_chain.run("train a random forest classifier")
        return result

# ==================== RAG (Retrieval Augmented Generation) ====================
class RAGSystem:
    """
    RAG system using LangChain for ML documentation Q&A
    """
    
    def __init__(self, api_key: str):
        self.llm = OpenAI(temperature=0, openai_api_key=api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vectorstore = None
    
    def build_knowledge_base(self, document_paths: List[str]):
        """Build vector store from documents"""
        documents = []
        
        # Load documents
        for path in document_paths:
            loader = TextLoader(path)
            documents.extend(loader.load())
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
    
    def create_qa_chain(self):
        """Create QA chain with retrieval"""
        if not self.vectorstore:
            raise ValueError("Knowledge base not built")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
        
        return qa_chain
    
    def ask_question(self, question: str):
        """Ask question about ML documentation"""
        qa_chain = self.create_qa_chain()
        result = qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }

# ==================== Agents ====================
class MLAgentSystem:
    """
    Agent system that can use tools to solve ML problems
    """
    
    def __init__(self, api_key: str):
        self.llm = OpenAI(temperature=0, openai_api_key=api_key)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def create_tools(self) -> List[Tool]:
        """Define tools for the agent"""
        
        def train_model(model_config: str) -> str:
            """Train ML model with given config"""
            # Simulate model training
            return f"Model trained with config: {model_config}"
        
        def evaluate_model(metrics: str) -> str:
            """Evaluate model performance"""
            # Simulate evaluation
            return f"Model evaluation: Accuracy=0.95, F1=0.93"
        
        def tune_hyperparameters(param_space: str) -> str:
            """Perform hyperparameter tuning"""
            # Simulate tuning
            return f"Best params: learning_rate=0.01, n_estimators=100"
        
        tools = [
            Tool(
                name="TrainModel",
                func=train_model,
                description="Train a machine learning model. Input should be model configuration."
            ),
            Tool(
                name="EvaluateModel",
                func=evaluate_model,
                description="Evaluate model performance. Input should be evaluation metrics."
            ),
            Tool(
                name="TuneHyperparameters",
                func=tune_hyperparameters,
                description="Tune model hyperparameters. Input should be parameter space."
            )
        ]
        
        return tools
    
    def create_agent(self):
        """Create conversational agent"""
        tools = self.create_tools()
        
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
        
        return agent
    
    def run_task(self, task: str):
        """Execute task using agent"""
        agent = self.create_agent()
        response = agent.run(task)
        return response

# ==================== Custom Chain for ML Pipeline ====================
from langchain.chains.base import Chain
from pydantic import BaseModel

class MLPipelineChain(Chain):
    """
    Custom chain for ML pipeline execution
    """
    
    llm: OpenAI
    
    @property
    def input_keys(self) -> List[str]:
        return ["dataset", "task_type"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["pipeline_steps", "estimated_time"]
    
    def _call(self, inputs: dict) -> dict:
        """Execute ML pipeline planning"""
        dataset = inputs["dataset"]
        task_type = inputs["task_type"]
        
        # Use LLM to plan pipeline
        prompt = f"""
        Plan an ML pipeline for:
        Dataset: {dataset}
        Task: {task_type}
        
        Provide:
        1. Data preprocessing steps
        2. Feature engineering
        3. Model selection
        4. Evaluation strategy
        """
        
        plan = self.llm(prompt)
        
        return {
            "pipeline_steps": plan,
            "estimated_time": "~2 hours"
        }

# ==================== Memory Systems ====================
class ConversationalMLAssistant:
    """ML assistant with memory"""
    
    def __init__(self, api_key: str):
        self.llm = OpenAI(temperature=0.7, openai_api_key=api_key)
        
        # Different memory types
        from langchain.memory import (
            ConversationBufferMemory,
            ConversationSummaryMemory,
            ConversationBufferWindowMemory
        )
        
        # Buffer memory: stores entire conversation
        self.buffer_memory = ConversationBufferMemory()
        
        # Window memory: stores last K interactions
        self.window_memory = ConversationBufferWindowMemory(k=5)
        
        # Summary memory: summarizes old conversations
        self.summary_memory = ConversationSummaryMemory(llm=self.llm)
    
    def chat_with_buffer_memory(self, user_input: str):
        """Chat with full conversation history"""
        from langchain.chains import ConversationChain
        
        conversation = ConversationChain(
            llm=self.llm,
            memory=self.buffer_memory,
            verbose=True
        )
        
        response = conversation.predict(input=user_input)
        return response

# ==================== Usage Examples ====================
def langchain_workflow():
    """Complete LangChain workflow"""
    api_key = "your-api-key"
    
    # 1. Simple chain
    simple = SimpleLangChainExample(api_key)
    response = simple.simple_chain()
    print("Simple chain:", response)
    
    # 2. RAG system
    rag = RAGSystem(api_key)
    rag.build_knowledge_base(["ml_docs.txt"])
    answer = rag.ask_question("How do I handle imbalanced datasets?")
    print("RAG answer:", answer)
    
    # 3. Agent system
    agent_system = MLAgentSystem(api_key)
    result = agent_system.run_task(
        "Train a random forest model and evaluate its performance"
    )
    print("Agent result:", result)
```

### Q7.2: What is LangGraph and when would you use it over LangChain?

**Answer:**

**LangGraph** is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with graph-based workflows.

**Key Differences:**

| Feature | LangChain | LangGraph |
|---------|-----------|-----------|
| Structure | Linear chains | Cyclical graphs |
| State Management | Limited | Built-in |
| Conditional Logic | Basic | Advanced |
| Use Case | Simple workflows | Complex workflows |

**When to use LangGraph:**
- Need cycles/loops in workflow
- Complex state management
- Multiple decision points
- Agent collaboration

**Analogy:** 
- **LangChain** = Recipe (linear steps)
- **LangGraph** = Flowchart (with branches, loops, conditions)

**Example:**
```python
from langgraph.graph import Graph, END
from typing import TypedDict, Annotated, Sequence
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import operator

# ==================== Basic LangGraph ====================
class GraphState(TypedDict):
    """State that gets passed between nodes"""
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    next_step: str
    iteration_count: int

class MLWorkflowGraph:
    """
    Multi-step ML workflow using LangGraph
    """
    
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
        self.graph = self.build_graph()
    
    def build_graph(self) -> Graph:
        """Build workflow graph"""
        workflow = Graph()
        
        # Define nodes (steps)
        workflow.add_node("analyze_data", self.analyze_data)
        workflow.add_node("select_model", self.select_model)
        workflow.add_node("train_model", self.train_model)
        workflow.add_node("evaluate_model", self.evaluate_model)
        workflow.add_node("tune_hyperparameters", self.tune_hyperparameters)
        
        # Define edges (flow)
        workflow.add_edge("analyze_data", "select_model")
        workflow.add_edge("select_model", "train_model")
        workflow.add_edge("train_model", "evaluate_model")
        
        # Conditional edge: if performance is poor, tune hyperparameters
        workflow.add_conditional_edges(
            "evaluate_model",
            self.should_tune,
            {
                "tune": "tune_hyperparameters",
                "end": END
            }
        )
        
        # Loop back after tuning
        workflow.add_edge("tune_hyperparameters", "train_model")
        
        # Set entry point
        workflow.set_entry_point("analyze_data")
        
        return workflow.compile()
    
    def analyze_data(self, state: GraphState) -> GraphState:
        """Analyze dataset"""
        print("Analyzing data...")
        
        prompt = "Analyze this dataset and suggest preprocessing steps"
        response = self.llm([HumanMessage(content=prompt)])
        
        state["messages"].append(response)
        state["next_step"] = "select_model"
        return state
    
    def select_model(self, state: GraphState) -> GraphState:
        """Select appropriate model"""
        print("Selecting model...")
        
        prompt = "Based on the data analysis, suggest the best ML model"
        response = self.llm([HumanMessage(content=prompt)])
        
        state["messages"].append(response)
        state["next_step"] = "train_model"
        return state
    
    def train_model(self, state: GraphState) -> GraphState:
        """Train the model"""
        print(f"Training model (iteration {state['iteration_count']})...")
        
        state["iteration_count"] += 1
        state["next_step"] = "evaluate_model"
        return state
    
    def evaluate_model(self, state: GraphState) -> GraphState:
        """Evaluate model performance"""
        print("Evaluating model...")
        
        # Simulate performance
        performance = 0.85 if state["iteration_count"] == 1 else 0.92
        
        message = f"Model performance: {performance}"
        state["messages"].append(AIMessage(content=message))
        
        return state
    
    def tune_hyperparameters(self, state: GraphState) -> GraphState:
        """Tune hyperparameters"""
        print("Tuning hyperparameters...")
        
        state["messages"].append(AIMessage(content="Hyperparameters tuned"))
        return state
    
    def should_tune(self, state: GraphState) -> str:
        """Decide whether to tune hyperparameters"""
        # Check last message for performance
        last_message = state["messages"][-1].content
        
        if "0.85" in last_message and state["iteration_count"] < 3:
            return "tune"
        return "end"
    
    def run(self, initial_input: str):
        """Execute the workflow"""
        initial_state = {
            "messages": [HumanMessage(content=initial_input)],
            "next_step": "analyze_data",
            "iteration_count": 0
        }
        
        result = self.graph.invoke(initial_state)
        return result

# ==================== Multi-Agent System ====================
class MultiAgentResearchSystem:
    """
    Multiple agents collaborating using LangGraph
    """
    
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(temperature=0.7, openai_api_key=api_key)
    
    def build_research_graph(self):
        """Build multi-agent research workflow"""
        workflow = Graph()
        
        # Define agent nodes
        workflow.add_node("researcher", self.research_agent)
        workflow.add_node("coder", self.code_agent)
        workflow.add_node("reviewer", self.review_agent)
        workflow.add_node("supervisor", self.supervisor_agent)
        
        # Supervisor decides which agent to call
        workflow.add_conditional_edges(
            "supervisor",
            self.route_to_agent,
            {
                "researcher": "researcher",
                "coder": "coder",
                "reviewer": "reviewer",
                "end": END
            }
        )
        
        # All agents report back to supervisor
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("coder", "supervisor")
        workflow.add_edge("reviewer", "supervisor")
        
        workflow.set_entry_point("supervisor")
        
        return workflow.compile()
    
    def supervisor_agent(self, state: GraphState) -> GraphState:
        """Supervisor decides next action"""
        prompt = f"""
        Given the conversation: {state['messages']}
        
        Decide which agent should act next:
        - researcher: Find information about ML topics
        - coder: Implement code
        - reviewer: Review and critique work
        - end: Task is complete
        
        Respond with just the agent name.
        """
        
        response = self.llm([HumanMessage(content=prompt)])
        state["next_step"] = response.content.lower().strip()
        
        return state
    
    def research_agent(self, state: GraphState) -> GraphState:
        """Research ML papers and concepts"""
        print("Researcher working...")
        
        prompt = "Research the latest techniques for this ML problem"
        response = self.llm([HumanMessage(content=prompt)])
        
        state["messages"].append(AIMessage(content=f"Researcher: {response.content}"))
        return state
    
    def code_agent(self, state: GraphState) -> GraphState:
        """Write implementation code"""
        print("Coder working...")
        
        prompt = "Implement the researched solution in Python"
        response = self.llm([HumanMessage(content=prompt)])
        
        state["messages"].append(AIMessage(content=f"Coder: {response.content}"))
        return state
    
    def review_agent(self, state: GraphState) -> GraphState:
        """Review the work"""
        print("Reviewer working...")
        
        prompt = "Review the implementation and suggest improvements"
        response = self.llm([HumanMessage(content=prompt)])
        
        state["messages"].append(AIMessage(content=f"Reviewer: {response.content}"))
        return state
    
    def route_to_agent(self, state: GraphState) -> str:
        """Route to next agent"""
        return state.get("next_step", "end")

# ==================== State Management ====================
class StatefulMLPipeline:
    """
    LangGraph for stateful ML pipeline with checkpoints
    """
    
    def __init__(self):
        self.checkpoints = {}
    
    def build_stateful_graph(self):
        """Build graph with state persistence"""
        workflow = Graph()
        
        workflow.add_node("load_data", self.load_data_node)
        workflow.add_node("preprocess", self.preprocess_node)
        workflow.add_node("train", self.train_node)
        workflow.add_node("validate", self.validate_node)
        
        # Add edges
        workflow.add_edge("load_data", "preprocess")
        workflow.add_edge("preprocess", "train")
        workflow.add_edge("train", "validate")
        
        # Conditional: retry training if validation fails
        workflow.add_conditional_edges(
            "validate",
            self.check_validation,
            {
                "retry": "train",
                "success": END
            }
        )
        
        workflow.set_entry_point("load_data")
        
        return workflow.compile()
    
    def load_data_node(self, state: dict) -> dict:
        """Load and checkpoint data"""
        print("Loading data...")
        state["data_loaded"] = True
        state["checkpoint"] = "data_loaded"
        self.checkpoints["data_loaded"] = state.copy()
        return state
    
    def preprocess_node(self, state: dict) -> dict:
        """Preprocess and checkpoint"""
        print("Preprocessing...")
        state["preprocessed"] = True
        state["checkpoint"] = "preprocessed"
        self.checkpoints["preprocessed"] = state.copy()
        return state
    
    def train_node(self, state: dict) -> dict:
        """Train model"""
        print("Training...")
        state["model_trained"] = True
        state["training_attempts"] = state.get("training_attempts", 0) + 1
        return state
    
    def validate_node(self, state: dict) -> dict:
        """Validate model"""
        print("Validating...")
        # Simulate validation
        state["validation_score"] = 0.90 if state["training_attempts"] >= 2 else 0.75
        return state
    
    def check_validation(self, state: dict) -> str:
        """Check if validation passed"""
        if state["validation_score"] < 0.85 and state["training_attempts"] < 3:
            return "retry"
        return "success"
    
    def resume_from_checkpoint(self, checkpoint_name: str):
        """Resume workflow from checkpoint"""
        if checkpoint_name in self.checkpoints:
            state = self.checkpoints[checkpoint_name]
            print(f"Resuming from checkpoint: {checkpoint_name}")
            return state
        return {}

# ==================== Usage ====================
def langgraph_examples():
    """LangGraph usage examples"""
    
    # Example 1: ML Workflow
    workflow = MLWorkflowGraph(api_key="your-key")
    result = workflow.run("Build a classification model for customer churn")
    print("Workflow result:", result)
    
    # Example 2: Multi-Agent System
    research_system = MultiAgentResearchSystem(api_key="your-key")
    graph = research_system.build_research_graph()
    
    initial_state = {
        "messages": [HumanMessage(content="Implement transformer attention mechanism")],
        "next_step": "supervisor"
    }
    result = graph.invoke(initial_state)
    
    # Example 3: Stateful Pipeline with Checkpoints
    pipeline = StatefulMLPipeline()
    graph = pipeline.build_stateful_graph()
    
    # Run pipeline
    result = graph.invoke({"training_attempts": 0})
    
    # Resume from checkpoint if needed
    state = pipeline.resume_from_checkpoint("preprocessed")
```

### Q7.3: What is CrewAI and how does it differ from other orchestration frameworks?

**Answer:**

**CrewAI** is a framework for orchestrating role-playing autonomous AI agents. Agents collaborate like a crew working together.

**Key Concepts:**
- **Agents:** Autonomous entities with specific roles
- **Tasks:** Assignments for agents
- **Tools:** Functions agents can use
- **Process:** How agents collaborate (sequential/hierarchical)

**Comparison:**

| Framework | Focus | Best For |
|-----------|-------|----------|
| LangChain | Chains & Tools | Simple workflows |
| LangGraph | Graph workflows | Complex state machines |
| CrewAI | Multi-agent collaboration | Team-based tasks |

**Analogy:** CrewAI is like managing a team where each member has specialized skills and they work together on a project.

**Example:**
```python
from crewai import Agent, Task, Crew, Process
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

# ==================== CrewAI for ML Development ====================
class MLDevelopmentCrew:
    """
    Multi-agent crew for ML project development
    """
    
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(temperature=0.7, openai_api_key=api_key)
    
    def create_agents(self):
        """Create specialized agents"""
        
        # Data Scientist Agent
        data_scientist = Agent(
            role="Data Scientist",
            goal="Analyze data and design ML solutions",
            backstory="""You are an expert data scientist with 10 years of experience.
            You excel at exploratory data analysis, feature engineering, and model selection.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # ML Engineer Agent
        ml_engineer = Agent(
            role="ML Engineer",
            goal="Implement and optimize ML models",
            backstory="""You are a senior ML engineer specializing in model implementation,
            training pipelines, and production deployment.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            tools=self.create_ml_tools()
        )
        
        # MLOps Engineer Agent
        mlops_engineer = Agent(
            role="MLOps Engineer",
            goal="Deploy and monitor ML models",
            backstory="""You are an MLOps expert who ensures models are production-ready,
            monitored, and maintainable.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # QA Engineer Agent
        qa_engineer = Agent(
            role="QA Engineer",
            goal="Test and validate ML systems",
            backstory="""You are a quality assurance specialist who ensures ML models
            meet performance requirements and are robust.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        return data_scientist, ml_engineer, mlops_engineer, qa_engineer
    
    def create_ml_tools(self):
        """Create tools for ML agents"""
        
        def train_model_tool(config: str) -> str:
            """Simulate model training"""
            return f"Model trained with config: {config}"
        
        def evaluate_model_tool(metrics: str) -> str:
            """Simulate model evaluation"""
            return "Accuracy: 0.95, F1: 0.93, AUC: 0.96"
        
        def deploy_model_tool(model_name: str) -> str:
            """Simulate model deployment"""
            return f"Model {model_name} deployed to production"
        
        tools = [
            Tool(
                name="TrainModel",
                func=train_model_tool,
                description="Train ML model with configuration"
            ),
            Tool(
                name="EvaluateModel",
                func=evaluate_model_tool,
                description="Evaluate model performance"
            ),
            Tool(
                name="DeployModel",
                func=deploy_model_tool,
                description="Deploy model to production"
            )
        ]
        
        return tools
    
    def create_tasks(self, agents):
        """Create tasks for the crew"""
        data_scientist, ml_engineer, mlops_engineer, qa_engineer = agents
        
        # Task 1: Data Analysis
        analyze_task = Task(
            description="""Analyze the customer churn dataset:
            1. Perform exploratory data analysis
            2. Identify important features
            3. Suggest preprocessing steps
            4. Recommend suitable models""",
            agent=data_scientist,
            expected_output="Comprehensive data analysis report with model recommendations"
        )
        
        # Task 2: Model Implementation
        implement_task = Task(
            description="""Based on the data analysis:
            1. Implement the recommended model
            2. Set up training pipeline
            3. Perform hyperparameter tuning
            4. Generate performance metrics""",
            agent=ml_engineer,
            expected_output="Trained model with performance metrics",
            context=[analyze_task]  # Depends on analysis
        )
        
        # Task 3: Testing & Validation
        test_task = Task(
            description="""Test the implemented model:
            1. Validate on test set
            2. Check for edge cases
            3. Verify performance requirements
            4. Document findings""",
            agent=qa_engineer,
            expected_output="QA test report with validation results",
            context=[implement_task]
        )
        
        # Task 4: Deployment
        deploy_task = Task(
            description="""Deploy the validated model:
            1. Set up production environment
            2. Configure monitoring
            3. Implement A/B testing
            4. Create rollback plan""",
            agent=mlops_engineer,
            expected_output="Deployment documentation and monitoring dashboard",
            context=[test_task]
        )
        
        return [analyze_task, implement_task, test_task, deploy_task]
    
    def run_project(self):
        """Execute ML project with crew"""
        # Create agents
        agents = self.create_agents()
        
        # Create tasks
        tasks = self.create_tasks(agents)
        
        # Create crew
        crew = Crew(
            agents=list(agents),
            tasks=tasks,
            process=Process.sequential,  # Tasks run in sequence
            verbose=2
        )
        
        # Execute project
        result = crew.kickoff()
        
        return result

# ==================== Hierarchical Process ====================
class HierarchicalMLCrew:
    """
    Crew with hierarchical management structure
    """
    
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(temperature=0.7, openai_api_key Dockerfile for ML application
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables
ENV MODEL_PATH=/app/models/model.pkl
ENV LOG_LEVEL=INFO

# Run application
CMD ["python", "app.py"]
```

### Q3.2: What is the difference between CMD and ENTRYPOINT in Docker?

**Answer:**
- **CMD:** Provides default arguments, can be overridden
- **ENTRYPOINT:** Defines the main executable, harder to override

**Analogy:** 
- **ENTRYPOINT** is like the main program (e.g., Python interpreter)
- **CMD** is like the default script to run (e.g., app.py)

**Example:**
```dockerfile
# Example 1: Using CMD only
FROM python:3.10-slim
COPY app.py .
CMD ["python", "app.py"]
# Run: docker run myimage  → runs python app.py
# Override: docker run myimage python other.py  → runs python other.py

# Example 2: Using ENTRYPOINT only
FROM python:3.10-slim
COPY app.py .
ENTRYPOINT ["python"]
CMD ["app.py"]
# Run: docker run myimage  → runs python app.py
# Override: docker run myimage other.py  → runs python other.py

# Example 3: ML inference server
FROM python:3.10-slim
WORKDIR /app
COPY . .

# ENTRYPOINT defines the executable
ENTRYPOINT ["python", "inference_server.py"]

# CMD provides default arguments
CMD ["--host", "0.0.0.0", "--port", "8000"]

# Run with defaults: docker run myimage
# Override arguments: docker run myimage --host localhost --port 9000
```

### Q3.3: How do you optimize Docker images for ML applications?

**Answer:**

**Key strategies:**
1. Use appropriate base images
2. Leverage layer caching
3. Multi-stage builds
4. Minimize layers
5. Use .dockerignore

**Example:**
```dockerfile
# .dockerignore file
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.git/
.gitignore
*.md
tests/
.pytest_cache/
.coverage
*.log
data/raw/  # Don't copy raw data
notebooks/
.vscode/
.idea/

# Optimized Multi-stage Dockerfile for ML
# Stage 1: Builder stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Create non-root user for security
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app
USER mluser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["python", "src/api/server.py"]
```

**Optimization Example with Size Comparison:**
```dockerfile
# Bad: Large image (~2GB)
FROM python:3.10
COPY . .
RUN pip install tensorflow torch transformers

# Good: Optimized image (~800MB)
FROM python:3.10-slim
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
```

### Q3.4: Explain Docker networking and how containers communicate.

**Answer:**

**Network Types:**
1. **Bridge:** Default, isolated network
2. **Host:** Uses host's network directly
3. **None:** No networking
4. **Custom:** User-defined networks

**Analogy:** Docker networks are like apartment buildings:
- **Bridge:** Apartments with private internal phones
- **Host:** Direct access to outside lines
- **Custom:** Private VLANs for specific groups

**Example:**
```yaml
# docker-compose.yml for ML microservices
version: '3.8'

services:
  # ML Model API
  model-api:
    build: ./model-service
    ports:
      - "8000:8000"
    networks:
      - ml-network
    environment:
      - MODEL_PATH=/models/model.pkl
      - DB_HOST=postgres
      - REDIS_HOST=redis
    volumes:
      - ./models:/models
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  # Feature Store
  feature-store:
    build: ./feature-service
    ports:
      - "8001:8001"
    networks:
      - ml-network
    environment:
      - DB_HOST=postgres
    depends_on:
      - postgres
  
  # Database
  postgres:
    image: postgres:14-alpine
    networks:
      - ml-network
    environment:
      - POSTGRES_DB=mldb
      - POSTGRES_USER=mluser
      - POSTGRES_PASSWORD=mlpassword
    volumes:
      - postgres-data:/var/lib/postgresql/data
  
  # Cache
  redis:
    image: redis:7-alpine
    networks:
      - ml-network
    volumes:
      - redis-data:/data

networks:
  ml-network:
    driver: bridge

volumes:
  postgres-data:
  redis-data:
```

**Communication Example:**
```python
# model-service/app.py
import requests
import redis
import psycopg2

class MLService:
    def __init__(self):
        # Connect to Redis cache (using service name as hostname)
        self.cache = redis.Redis(host='redis', port=6379)
        
        # Connect to PostgreSQL (using service name)
        self.db = psycopg2.connect(
            host='postgres',
            database='mldb',
            user='mluser',
            password='mlpassword'
        )
    
    def get_features(self, user_id: str):
        # Check cache first
        cached = self.cache.get(f"features:{user_id}")
        if cached:
            return cached
        
        # Call feature-store service
        response = requests.get(
            f"http://feature-store:8001/features/{user_id}"
        )
        features = response.json()
        
        # Cache result
        self.cache.setex(f"features:{user_id}", 3600, str(features))
        return features
```

### Q3.5: How do you handle secrets and sensitive data in Docker?

**Answer:**

**Best practices:**
1. Never hardcode secrets in Dockerfiles
2. Use environment variables
3. Use Docker secrets (Swarm) or external secret managers
4. Mount secrets as files
5. Use .env files (not in version control)
