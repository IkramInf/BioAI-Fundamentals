# Advanced Python Topics - Complete Guide

## 1. Concurrency and Parallelism

**Explanation:** Concurrency means handling multiple tasks by switching between them (useful for I/O-bound tasks), while parallelism means executing multiple tasks simultaneously on multiple CPU cores (useful for CPU-bound tasks).

**Use Cases:** Web scraping, API calls, data processing, scientific computing

**Code Example:**
```python
# Concurrent execution (switching between tasks)
import time

def task(name, delay):
    print(f"Task {name} starting")
    time.sleep(delay)
    print(f"Task {name} completed")

# Sequential
start = time.time()
task("A", 1)
task("B", 1)
print(f"Sequential time: {time.time() - start:.2f}s")  # ~2 seconds
```

---

## 2. Threading and ThreadPoolExecutor

**Explanation:** Threading allows multiple threads to run concurrently within a single process. ThreadPoolExecutor manages a pool of worker threads for concurrent execution. Best for I/O-bound tasks due to Python's Global Interpreter Lock (GIL).

**Use Cases:** Network requests, file I/O, database queries, API calls

**Code Example:**
```python
from concurrent.futures import ThreadPoolExecutor
import time

def fetch_data(url):
    time.sleep(1)  # Simulate network request
    return f"Data from {url}"

urls = ["url1", "url2", "url3", "url4"]

# Using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(fetch_data, urls)
    
for result in results:
    print(result)  # Completes in ~1 second instead of 4
```

---

## 3. Multiprocessing and ProcessPoolExecutor

**Explanation:** Multiprocessing creates separate Python processes, each with its own interpreter and memory space, bypassing the GIL. ProcessPoolExecutor manages a pool of worker processes for parallel execution.

**Use Cases:** CPU-intensive computations, image processing, data analysis, scientific computing

**Code Example:**
```python
from concurrent.futures import ProcessPoolExecutor
import math

def compute_factorial(n):
    return math.factorial(n)

numbers = [5000, 6000, 7000, 8000]

# Using ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=4) as executor:
    results = executor.map(compute_factorial, numbers)
    
for num, result in zip(numbers, results):
    print(f"Factorial of {num} has {len(str(result))} digits")
```

---

## 4. Asyncio and Asynchronous Programming

**Explanation:** Asyncio enables cooperative multitasking using async/await syntax. It's single-threaded but allows handling many I/O operations concurrently by yielding control during waiting periods.

**Use Cases:** Web servers, websockets, concurrent API requests, real-time applications

**Code Example:**
```python
import asyncio
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = [
        'https://api.github.com',
        'https://httpbin.org/get',
        'https://jsonplaceholder.typicode.com/posts/1'
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

# Run the async function
# results = asyncio.run(main())
```

---

## 5. Decorators and Higher-Order Functions

**Explanation:** Decorators are functions that modify or enhance other functions. Higher-order functions take functions as arguments or return functions.

**Use Cases:** Logging, authentication, caching, timing, validation

**Code Example:**
```python
import time
from functools import wraps

# Decorator for timing functions
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.4f}s")
        return result
    return wrapper

# Decorator with arguments
def retry(max_attempts=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}")
        return wrapper
    return decorator

@timer
@retry(max_attempts=3)
def fetch_data():
    # Simulate operation
    time.sleep(0.5)
    return "Data fetched"

result = fetch_data()
```

---

## 6. Context Managers (`with` statement, `__enter__`, `__exit__`)

**Explanation:** Context managers handle resource setup and cleanup automatically. The `with` statement ensures proper acquisition and release of resources.

**Use Cases:** File handling, database connections, locks, temporary state changes

**Code Example:**
```python
from contextlib import contextmanager

# Using class-based context manager
class DatabaseConnection:
    def __enter__(self):
        print("Opening database connection")
        self.connection = "DB Connection"
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection")
        if exc_type:
            print(f"Exception occurred: {exc_val}")
        return False  # Don't suppress exceptions

with DatabaseConnection() as conn:
    print(f"Using {conn}")

# Using contextmanager decorator
@contextmanager
def temporary_setting(key, value):
    old_value = globals().get(key)
    globals()[key] = value
    try:
        yield
    finally:
        if old_value is None:
            del globals()[key]
        else:
            globals()[key] = old_value

with temporary_setting('DEBUG', True):
    print(f"DEBUG: {globals().get('DEBUG')}")
```

---

## 7. Design Patterns (Factory, Singleton, Strategy, Observer, etc.)

**Explanation:** Design patterns are reusable solutions to common software design problems.

**Use Cases:** Building scalable applications, improving code maintainability

**Code Example:**
```python
# Singleton Pattern
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Factory Pattern
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        raise ValueError(f"Unknown animal type: {animal_type}")

# Strategy Pattern
class PaymentStrategy:
    def pay(self, amount):
        pass

class CreditCardPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid ${amount} with credit card"

class PayPalPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid ${amount} with PayPal"

class ShoppingCart:
    def __init__(self, payment_strategy):
        self.payment_strategy = payment_strategy
    
    def checkout(self, amount):
        return self.payment_strategy.pay(amount)

cart = ShoppingCart(CreditCardPayment())
print(cart.checkout(100))
```

---

## 8. Mixins and Multiple Inheritance

**Explanation:** Mixins are classes that provide methods to other classes through inheritance without being base classes themselves. Multiple inheritance allows a class to inherit from multiple parent classes.

**Use Cases:** Adding functionality to classes, composition over inheritance

**Code Example:**
```python
# Mixin classes
class JSONMixin:
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

class TimestampMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from datetime import datetime
        self.created_at = datetime.now()

# Using mixins
class User(JSONMixin, TimestampMixin):
    def __init__(self, name, email):
        super().__init__()
        self.name = name
        self.email = email

user = User("Alice", "alice@example.com")
print(user.to_json())
print(f"Created at: {user.created_at}")
```

---

## 9. Descriptors and Property Management

**Explanation:** Descriptors are objects that define how attribute access is interpreted. They implement `__get__`, `__set__`, and/or `__delete__` methods. Properties are a convenient way to use descriptors.

**Use Cases:** Validation, computed attributes, lazy loading, type checking

**Code Example:**
```python
# Descriptor for validation
class PositiveNumber:
    def __init__(self, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name, 0)
    
    def __set__(self, obj, value):
        if value < 0:
            raise ValueError(f"{self.name} must be positive")
        obj.__dict__[self.name] = value

class BankAccount:
    balance = PositiveNumber('balance')
    
    def __init__(self, balance):
        self.balance = balance

# Using property decorator
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32

temp = Temperature(25)
print(f"{temp.celsius}°C = {temp.fahrenheit}°F")
```

---

## 10. Metaclasses and Class Customization

**Explanation:** Metaclasses are classes of classes. They define how classes behave and are created. The default metaclass is `type`.

**Use Cases:** ORM frameworks, API frameworks, enforcing coding standards, automatic registration

**Code Example:**
```python
# Simple metaclass
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        print("Initializing database")

db1 = Database()
db2 = Database()  # Same instance
print(db1 is db2)  # True

# Automatic registration metaclass
class RegisteredMeta(type):
    registry = {}
    
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        if name != 'Plugin':  # Don't register base class
            mcs.registry[name] = cls
        return cls

class Plugin(metaclass=RegisteredMeta):
    pass

class PDFPlugin(Plugin):
    pass

class CSVPlugin(Plugin):
    pass

print(RegisteredMeta.registry)  # {'PDFPlugin': ..., 'CSVPlugin': ...}
```

---

## 11. Type Hints and Static Typing (`typing`, `mypy`)

**Explanation:** Type hints provide optional static typing to Python. They improve code clarity and enable static type checkers like mypy to catch type errors.

**Use Cases:** Large codebases, API development, documentation, IDE support

**Code Example:**
```python
from typing import List, Dict, Optional, Union, Callable, TypeVar, Generic

def greet(name: str) -> str:
    return f"Hello, {name}"

def process_items(items: List[int]) -> Dict[str, int]:
    return {
        'count': len(items),
        'sum': sum(items),
        'max': max(items) if items else 0
    }

# Optional and Union types
def find_user(user_id: int) -> Optional[Dict[str, str]]:
    # Returns None if user not found
    return {'name': 'Alice', 'email': 'alice@example.com'}

def parse_value(value: Union[int, str]) -> int:
    return int(value)

# Generic types
T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self):
        self.items: List[T] = []
    
    def push(self, item: T) -> None:
        self.items.append(item)
    
    def pop(self) -> T:
        return self.items.pop()

stack: Stack[int] = Stack()
stack.push(1)
stack.push(2)

# Callable types
def apply_operation(x: int, y: int, operation: Callable[[int, int], int]) -> int:
    return operation(x, y)

result = apply_operation(5, 3, lambda a, b: a + b)
```

---

## 12. Functional Programming (`map`, `reduce`, `partial`, `lambda`, `functools`)

**Explanation:** Functional programming treats computation as the evaluation of mathematical functions, avoiding changing state and mutable data.

**Use Cases:** Data transformation, pipeline processing, clean code

**Code Example:**
```python
from functools import reduce, partial

# map, filter, lambda
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

# reduce
product = reduce(lambda x, y: x * y, numbers)  # 120

# partial application
def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(5))    # 125

# Function composition
def compose(*functions):
    def inner(arg):
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result
    return inner

add_10 = lambda x: x + 10
multiply_2 = lambda x: x * 2
subtract_5 = lambda x: x - 5

pipeline = compose(subtract_5, multiply_2, add_10)
print(pipeline(5))  # (5 + 10) * 2 - 5 = 25
```

---

## 13. Caching and Memoization (`functools.lru_cache`)

**Explanation:** Caching stores results of expensive function calls and returns cached result when same inputs occur again.

**Use Cases:** Recursive functions, expensive computations, API calls

**Code Example:**
```python
from functools import lru_cache, cache
import time

# Without cache
def fibonacci_slow(n):
    if n < 2:
        return n
    return fibonacci_slow(n-1) + fibonacci_slow(n-2)

# With LRU cache (Least Recently Used)
@lru_cache(maxsize=128)
def fibonacci_fast(n):
    if n < 2:
        return n
    return fibonacci_fast(n-1) + fibonacci_fast(n-2)

# Python 3.9+ has unbounded cache decorator
@cache
def expensive_function(x, y):
    time.sleep(1)
    return x + y

# First call takes 1 second
start = time.time()
result1 = expensive_function(5, 3)
print(f"First call: {time.time() - start:.2f}s")

# Second call is instant (cached)
start = time.time()
result2 = expensive_function(5, 3)
print(f"Second call: {time.time() - start:.4f}s")

# Check cache stats
print(fibonacci_fast.cache_info())

# Clear cache
fibonacci_fast.cache_clear()
```

---

## 14. Generators and Iterators

**Explanation:** Generators are functions that yield values one at a time, maintaining their state between yields. They're memory-efficient for large datasets.

**Use Cases:** Processing large files, infinite sequences, pipeline processing

**Code Example:**
```python
# Generator function
def count_up_to(n):
    count = 1
    while count <= n:
        yield count
        count += 1

for num in count_up_to(5):
    print(num)

# Generator expression
squares = (x**2 for x in range(10))

# Reading large files efficiently
def read_large_file(file_path):
    with open(file_path) as f:
        for line in f:
            yield line.strip()

# Generator pipeline
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def take(n, iterable):
    for i, item in enumerate(iterable):
        if i >= n:
            break
        yield item

# Get first 10 fibonacci numbers
fib_numbers = list(take(10, fibonacci()))
print(fib_numbers)

# Custom iterator class
class Countdown:
    def __init__(self, start):
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

for num in Countdown(5):
    print(num)
```

---

## 15. Coroutines and Event Loops

**Explanation:** Coroutines are special functions that can pause and resume execution. Event loops schedule and run asynchronous tasks.

**Use Cases:** Concurrent I/O operations, real-time systems, game loops

**Code Example:**
```python
import asyncio

# Basic coroutine
async def say_hello(name, delay):
    await asyncio.sleep(delay)
    print(f"Hello, {name}!")
    return f"Greeted {name}"

# Event loop with multiple tasks
async def main():
    # Run concurrently
    results = await asyncio.gather(
        say_hello("Alice", 1),
        say_hello("Bob", 2),
        say_hello("Charlie", 1)
    )
    print(results)

# asyncio.run(main())

# Producer-consumer pattern
async def producer(queue, n):
    for i in range(n):
        await asyncio.sleep(0.1)
        await queue.put(i)
        print(f"Produced: {i}")
    await queue.put(None)  # Sentinel

async def consumer(queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        await asyncio.sleep(0.2)
        print(f"Consumed: {item}")

async def run_producer_consumer():
    queue = asyncio.Queue()
    await asyncio.gather(
        producer(queue, 5),
        consumer(queue)
    )

# asyncio.run(run_producer_consumer())

# Timeout handling
async def long_operation():
    await asyncio.sleep(5)
    return "Done"

async def with_timeout():
    try:
        result = await asyncio.wait_for(long_operation(), timeout=2.0)
    except asyncio.TimeoutError:
        print("Operation timed out")

# asyncio.run(with_timeout())
```

---

## 16. Vectorization and Broadcasting (NumPy)

**Explanation:** Vectorization performs operations on entire arrays at once rather than using loops. Broadcasting allows operations on arrays of different shapes.

**Use Cases:** Scientific computing, data analysis, machine learning

**Code Example:**
```python
import numpy as np

# Vectorization - much faster than loops
arr = np.array([1, 2, 3, 4, 5])
squared = arr ** 2  # Vectorized operation

# Compare with loop (slow)
squared_loop = [x**2 for x in arr]

# Broadcasting
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
vector = np.array([10, 20, 30])

# Broadcasting adds vector to each row
result = matrix + vector
# [[11, 22, 33],
#  [14, 25, 36]]

# Universal functions (ufuncs)
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Element-wise operations
print(np.add(a, b))
print(np.multiply(a, b))
print(np.maximum(a, 2))

# Advanced indexing and filtering
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
mask = data > 5
filtered = data[mask]  # [6, 7, 8, 9]

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

dot_product = np.dot(A, B)
element_wise = A * B
```

---

## 17. Memory Optimization and Profiling

**Explanation:** Memory profiling identifies memory usage patterns and helps optimize memory consumption in applications.

**Use Cases:** Working with large datasets, memory-constrained environments

**Code Example:**
```python
import sys
from array import array

# Memory-efficient data structures
# List vs array
list_data = [1, 2, 3, 4, 5] * 1000
array_data = array('i', [1, 2, 3, 4, 5] * 1000)

print(f"List size: {sys.getsizeof(list_data)} bytes")
print(f"Array size: {sys.getsizeof(array_data)} bytes")

# Generators for memory efficiency
def process_large_data(n):
    # Generator - memory efficient
    for i in range(n):
        yield i ** 2

# Using __slots__ to reduce memory
class RegularClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class OptimizedClass:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

regular = RegularClass(1, 2)
optimized = OptimizedClass(1, 2)

print(f"Regular class: {sys.getsizeof(regular)} bytes")
print(f"Optimized class: {sys.getsizeof(optimized)} bytes")

# Memory profiling with tracemalloc
import tracemalloc

tracemalloc.start()

# Code to profile
data = [i for i in range(1000000)]

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

tracemalloc.stop()
```

---

## 18. Logging and Exception Handling Best Practices

**Explanation:** Proper logging and exception handling make applications maintainable and debuggable.

**Use Cases:** Production applications, debugging, monitoring

**Code Example:**
```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Rotating file handler
handler = RotatingFileHandler(
    'app.log',
    maxBytes=1024*1024,  # 1MB
    backupCount=5
)
logger.addHandler(handler)

# Custom exception classes
class ValidationError(Exception):
    """Raised when validation fails"""
    pass

class DatabaseError(Exception):
    """Raised when database operation fails"""
    pass

# Exception handling best practices
def process_user_data(user_data):
    try:
        if not user_data.get('email'):
            raise ValidationError("Email is required")
        
        # Simulate database operation
        result = save_to_database(user_data)
        logger.info(f"User {user_data['email']} saved successfully")
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise  # Re-raise for caller to handle
        
    except DatabaseError as e:
        logger.critical(f"Database error: {e}")
        # Handle or raise
        
    except Exception as e:
        logger.exception("Unexpected error occurred")
        raise
        
    finally:
        logger.debug("Cleanup operations")

def save_to_database(data):
    # Simulated function
    pass

# Context manager for exception handling
from contextlib import suppress

# Suppress specific exceptions
with suppress(FileNotFoundError):
    with open('nonexistent.txt') as f:
        content = f.read()

# Logging different levels
logger.debug("Detailed information for debugging")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical error")
```

---

## 19. Testing and Mocking (pytest, unittest)

**Explanation:** Testing ensures code correctness. Mocking isolates units of code by replacing dependencies with controlled substitutes.

**Use Cases:** Unit testing, integration testing, TDD

**Code Example:**
```python
import pytest
from unittest.mock import Mock, patch, MagicMock

# Code to test
class UserService:
    def __init__(self, database):
        self.database = database
    
    def get_user(self, user_id):
        user = self.database.find_user(user_id)
        if not user:
            raise ValueError("User not found")
        return user
    
    def create_user(self, name, email):
        if self.database.user_exists(email):
            raise ValueError("User already exists")
        return self.database.create(name, email)

# pytest tests
class TestUserService:
    def test_get_user_success(self):
        # Mock database
        mock_db = Mock()
        mock_db.find_user.return_value = {'id': 1, 'name': 'Alice'}
        
        service = UserService(mock_db)
        user = service.get_user(1)
        
        assert user['name'] == 'Alice'
        mock_db.find_user.assert_called_once_with(1)
    
    def test_get_user_not_found(self):
        mock_db = Mock()
        mock_db.find_user.return_value = None
        
        service = UserService(mock_db)
        
        with pytest.raises(ValueError, match="User not found"):
            service.get_user(999)
    
    def test_create_user_already_exists(self):
        mock_db = Mock()
        mock_db.user_exists.return_value = True
        
        service = UserService(mock_db)
        
        with pytest.raises(ValueError, match="already exists"):
            service.create_user("Bob", "bob@example.com")

# Fixtures
@pytest.fixture
def user_service():
    mock_db = Mock()
    return UserService(mock_db)

def test_with_fixture(user_service):
    assert user_service.database is not None

# Parametrized tests
@pytest.mark.parametrize("user_id,expected", [
    (1, {'id': 1, 'name': 'Alice'}),
    (2, {'id': 2, 'name': 'Bob'}),
])
def test_multiple_users(user_id, expected):
    mock_db = Mock()
    mock_db.find_user.return_value = expected
    
    service = UserService(mock_db)
    result = service.get_user(user_id)
    
    assert result == expected

# Patching
@patch('module_name.external_api_call')
def test_with_patch(mock_api):
    mock_api.return_value = {'status': 'success'}
    # Test code that calls external_api_call
```

---

## 20. Packaging and Dependency Management (`poetry`, `pipenv`)

**Explanation:** Modern Python packaging tools manage dependencies, virtual environments, and package distribution.

**Use Cases:** Project setup, dependency isolation, package publishing

**Code Example:**
```bash
# Poetry setup
poetry init
poetry add requests
poetry add --dev pytest
poetry install
poetry run python app.py
poetry build
poetry publish

# pyproject.toml
[tool.poetry]
name = "myproject"
version = "0.1.0"
description = "My awesome project"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.9"
requests = "^2.28.0"
pandas = "^1.5.0"

[tool.poetry.dev-dependencies]
pytest = "^7.2.0"
black = "^22.10.0"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

```python
# setup.py for traditional packaging
from setuptools import setup, find_packages

setup(
    name='myproject',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'requests>=2.28.0',
        'pandas>=1.5.0',
    ],
    extras_require={
        'dev': ['pytest>=7.2.0', 'black>=22.10.0'],
    },
    entry_points={
        'console_scripts': [
            'myapp=myproject.cli:main',
        ],
    },
)
```

---

## 21. Advanced File I/O and Memory-Mapped Files

**Explanation:** Memory-mapped files allow treating file content as if it's in memory, enabling efficient access to large files.

**Use Cases:** Large file processing, shared memory, binary data

**Code Example:**
```python
import mmap
import os

# Memory-mapped file
with open('large_file.dat', 'r+b') as f:
    mmapped_file = mmap.mmap(f.fileno(), 0)
    
    # Read bytes
    data = mmapped_file[:100]
    
    # Search in file
    index = mmapped_file.find(b'search_term')
    
    # Modify file
    mmapped_file[0:5] = b'Hello'
    
    mmapped_file.close()

# Binary file operations
import struct

# Write binary data
with open('data.bin', 'wb') as f:
    # Write integers
    f.write(struct.pack('i', 42))
    f.write(struct.pack('f', 3.14))
    
    # Write multiple values
    data = struct.pack('iif', 1, 2, 3.5)
    f.write(data)

# Read binary data
with open('data.bin', 'rb') as f:
    # Read integer
    value = struct.unpack('i', f.read(4))[0]
    
    # Read float
    pi = struct.unpack('f', f.read(4))[0]

# Efficient line processing
def process_large_file(filename):
    with open(filename, 'r') as f:
        # Read in chunks
        chunk_size = 1024 * 1024  # 1MB
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            process_chunk(chunk)

def process_chunk(chunk):
    # Process the chunk
    pass

# File locking
import fcntl

def write_with_lock(filename, data):
    with open(filename, 'a') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(data)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

---

## 22. Serialization and Pickling

**Explanation:** Serialization converts Python objects to byte streams for storage or transmission. Pickle is Python's native serialization format.

**Use Cases:** Object persistence, caching, inter-process communication

**Code Example:**
```python
import pickle
import json
import marshal

# Pickle - serialize Python objects
data = {
    'name': 'Alice',
    'scores': [95, 87, 92],
    'metadata': {'age': 30}
}

# Serialize to file
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)

# Deserialize from file
with open('data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# Serialize to bytes
serialized = pickle.dumps(data)
deserialized = pickle.loads(serialized)

# Custom class serialization
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __getstate__(self):
        # Custom serialization logic
        state = self.__dict__.copy()
        # Don't pickle sensitive data
        state.pop('_cached_data', None)
        return state
    
    def __setstate__(self, state):
        # Custom deserialization logic
        self.__dict__.update(state)

person = Person("Bob", 25)
pickled_person = pickle.dumps(person)
restored_person = pickle.loads(pickled_person)

# JSON serialization (cross-platform)
json_str = json.dumps(data)
loaded_json = json.loads(json_str)

# Custom JSON encoder
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

data_with_date = {
    'timestamp': datetime.now(),
    'value': 42
}

json_str = json.dumps(data_with_date, cls=DateTimeEncoder)

# Marshal (faster but less flexible)
marshal_data = marshal.dumps([1, 2, 3, 'test'])
loaded_marshal = marshal.loads(marshal_data)
```

---

## 23. Parallel Data Processing (Dask, Joblib, Ray)

**Explanation:** Frameworks for parallel and distributed computing that scale beyond single-machine limitations.

**Use Cases:** Big data processing, distributed computing, parallel machine learning

**Code Example:**
```python
# Joblib for simple parallelization
from joblib import Parallel, delayed

def process_item(item):
    return item ** 2

# Parallel processing
results = Parallel(n_jobs=4)(
    delayed(process_item)(i) for i in range(100)
)

# Dask for larger-than-memory computations
import dask.dataframe as dd
import dask.array as da

# Dask DataFrame (like pandas but parallel)
# df = dd.read_csv('large_file.csv')
# result = df.groupby('category').mean().compute()

# Dask Array (like NumPy but parallel)
x = da.random.random((10000, 10000), chunks=(1000, 1000))
y = x + x.T
result = y.mean().compute()

# Ray for distributed computing
import ray

# ray.init()

@ray.remote
def expensive_function(x):
    # Simulate expensive computation
    return x ** 2

# Parallel execution with Ray
# futures = [expensive_function.remote(i) for i in range(10)]
# results = ray.get(futures)

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1
        return self.value

# Distributed actors
# counter = Counter.remote()
# ray.get(counter.increment.remote())
```

---

## 24. GPU Acceleration (Numba, CuPy, PyTorch, TensorFlow)

**Explanation:** Leverage GPU computing for massive parallel processing speedups on numerical computations.

**Use Cases:** Deep learning, scientific computing, image processing

**Code Example:**
```python
# Numba JIT compilation
from numba import jit, cuda
import numpy as np

@jit(nopython=True)
def fast_function(x):
    total = 0
    for i in range(x.shape[0]):
        total += x[i] ** 2
    return total

data = np.arange(1000000)
result = fast_function(data)

# Numba parallel
from numba import prange

@jit(parallel=True)
def parallel_sum(x):
    total = 0
    for i in prange(x.shape[0]):
        total += x[i]
    return total

# CuPy (NumPy-like GPU arrays)
import cupy as cp

# GPU array
x_gpu = cp.array([1, 2, 3, 4, 5])
y_gpu = cp.array([10, 20, 30, 40, 50])

# GPU computation
result_gpu = cp.dot(x_gpu, y_gpu)

# Transfer back to CPU
result_cpu = cp.asnumpy(result_gpu)

# PyTorch GPU acceleration
import torch

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    
    # Create tensor on GPU
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)
    
    # GPU computation
    z = x + y
    
    # Move to CPU
    z_cpu = z.cpu()

# Custom CUDA kernel with Numba
@cuda.jit
def gpu_add(x, y, out):
    idx = cuda.grid(1)
    if idx < x.shape[0]:
        out[idx] = x[idx] + y[idx]

# Allocate arrays
x = np.arange(1000000)
y = np.arange(1000000)
out = np.zeros_like(x)

# Configure and launch kernel
threads_per_block = 256
blocks_per_grid = (x.size + threads_per_block - 1) // threads_per_block
# gpu_add[blocks_per_grid, threads_per_block](x, y, out)
```

---

## 25. Dependency Injection and Inversion of Control

**Explanation:** Design pattern where objects receive dependencies rather than creating them, improving testability and flexibility.

**Use Cases:** Large applications, testing, plugin systems

**Code Example:**
```python
from abc import ABC, abstractmethod
from typing import Protocol

# Protocol-based dependency injection
class DatabaseProtocol(Protocol):
    def query(self, sql: str): ...
    def execute(self, sql: str): ...

class PostgresDatabase:
    def query(self, sql: str):
        return f"Postgres query: {sql}"
    
    def execute(self, sql: str):
        return f"Postgres execute: {sql}"

class MySQLDatabase:
    def query(self, sql: str):
        return f"MySQL query: {sql}"
    
    def execute(self, sql: str):
        return f"MySQL execute: {sql}"

# Service depends on abstraction, not concrete implementation
class UserService:
    def __init__(self, database: DatabaseProtocol):
        self.database = database
    
    def get_user(self, user_id: int):
        return self.database.query(f"SELECT * FROM users WHERE id={user_id}")

# Dependency injection
postgres = PostgresDatabase()
mysql = MySQLDatabase()

service1 = UserService(postgres)  # Inject Postgres
service2 = UserService(mysql)     # Inject MySQL

# Simple DI Container
class DIContainer:
    def __init__(self):
        self._services = {}
    
    def register(self, interface, implementation):
        self._services[interface] = implementation
    
    def resolve(self, interface):
        implementation = self._services.get(interface)
        if not implementation:
            raise ValueError(f"No implementation for {interface}")
        
        # Auto-inject dependencies
        if hasattr(implementation, '__init__'):
            return implementation()
        return implementation

container = DIContainer()
container.register('database', PostgresDatabase)
container.register('user_service', lambda: UserService(container.resolve('database')))

# Factory pattern with DI
class ServiceFactory:
    def __init__(self):
        self.database = PostgresDatabase()
    
    def create_user_service(self):
        return UserService(self.database)
    
    def create_order_service(self):
        return OrderService(self.database)

class OrderService:
    def __init__(self, database: DatabaseProtocol):
        self.database = database
```

---

## 26. Asynchronous Web Frameworks (FastAPI, aiohttp)

**Explanation:** Async web frameworks handle many concurrent connections efficiently using asyncio.

**Use Cases:** High-performance APIs, real-time applications, microservices

**Code Example:**
```python
# FastAPI example
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    description: str = None

# In-memory database
items_db = []

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id >= len(items_db):
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]

@app.post("/items/")
async def create_item(item: Item):
    items_db.append(item)
    return {"id": len(items_db) - 1, "item": item}

@app.get("/items/", response_model=List[Item])
async def list_items(skip: int = 0, limit: int = 10):
    return items_db[skip:skip + limit]

# Dependency injection
async def get_current_user():
    # Simulate user authentication
    return {"username": "testuser"}

@app.get("/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user

# Background tasks
from fastapi import BackgroundTasks

def send_email(email: str, message: str):
    print(f"Sending email to {email}: {message}")

@app.post("/send-notification/")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(send_email, email, "Hello!")
    return {"message": "Notification sent in background"}

# aiohttp example
import aiohttp
from aiohttp import web

async def handle_get(request):
    return web.json_response({"message": "Hello"})

async def handle_post(request):
    data = await request.json()
    return web.json_response({"received": data})

# aiohttp app
aiohttp_app = web.Application()
aiohttp_app.router.add_get('/', handle_get)
aiohttp_app.router.add_post('/data', handle_post)

# web.run_app(aiohttp_app, port=8080)
```

---

## 27. Modular and Plugin-Based Architectures

**Explanation:** Design applications to load functionality dynamically through plugins, enabling extensibility without modifying core code.

**Use Cases:** Extensible applications, plugin systems, modular frameworks

**Code Example:**
```python
import importlib
import os
from abc import ABC, abstractmethod
from pathlib import Path

# Plugin interface
class Plugin(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

# Plugin manager
class PluginManager:
    def __init__(self, plugin_dir='plugins'):
        self.plugins = {}
        self.plugin_dir = plugin_dir
    
    def discover_plugins(self):
        """Automatically discover and load plugins"""
        plugin_path = Path(self.plugin_dir)
        if not plugin_path.exists():
            return
        
        for file in plugin_path.glob('*.py'):
            if file.name.startswith('_'):
                continue
            
            module_name = file.stem
            spec = importlib.util.spec_from_file_location(module_name, file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find Plugin subclasses
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, Plugin) and 
                    attr is not Plugin):
                    self.register_plugin(attr())
    
    def register_plugin(self, plugin: Plugin):
        self.plugins[plugin.get_name()] = plugin
    
    def get_plugin(self, name: str) -> Plugin:
        return self.plugins.get(name)
    
    def list_plugins(self):
        return list(self.plugins.keys())
    
    def execute_plugin(self, name: str, *args, **kwargs):
        plugin = self.get_plugin(name)
        if not plugin:
            raise ValueError(f"Plugin '{name}' not found")
        return plugin.execute(*args, **kwargs)

# Example plugins
class PDFExportPlugin(Plugin):
    def get_name(self) -> str:
        return "pdf_export"
    
    def execute(self, data):
        return f"Exporting {data} to PDF"

class CSVExportPlugin(Plugin):
    def get_name(self) -> str:
        return "csv_export"
    
    def execute(self, data):
        return f"Exporting {data} to CSV"

# Usage
manager = PluginManager()
manager.register_plugin(PDFExportPlugin())
manager.register_plugin(CSVExportPlugin())

# Execute plugins
result = manager.execute_plugin("pdf_export", "report.pdf")
print(manager.list_plugins())

# Hook-based plugin system
class HookManager:
    def __init__(self):
        self.hooks = {}
    
    def register_hook(self, hook_name, callback):
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)
    
    def execute_hook(self, hook_name, *args, **kwargs):
        if hook_name in self.hooks:
            results = []
            for callback in self.hooks[hook_name]:
                results.append(callback(*args, **kwargs))
            return results
        return []

hooks = HookManager()
hooks.register_hook('before_save', lambda data: print(f"Validating {data}"))
hooks.register_hook('after_save', lambda data: print(f"Saved {data}"))

hooks.execute_hook('before_save', 'user_data')
```

---

## 28. Context-Aware Resource Management

**Explanation:** Advanced patterns for managing resources with awareness of execution context, enabling proper cleanup and resource sharing.

**Use Cases:** Database connections, file handles, network connections, transaction management

**Code Example:**
```python
from contextlib import contextmanager, ExitStack
import threading
from typing import Optional

# Thread-local storage for context
class ContextManager:
    def __init__(self):
        self._local = threading.local()
    
    def set_context(self, key, value):
        if not hasattr(self._local, 'context'):
            self._local.context = {}
        self._local.context[key] = value
    
    def get_context(self, key):
        if hasattr(self._local, 'context'):
            return self._local.context.get(key)
        return None
    
    def clear_context(self):
        if hasattr(self._local, 'context'):
            self._local.context.clear()

context_manager = ContextManager()

# Context-aware database connection
class DatabaseConnection:
    _instance: Optional['DatabaseConnection'] = None
    
    def __init__(self):
        self.connection = None
        self.transaction_depth = 0
    
    def connect(self):
        if not self.connection:
            self.connection = "DB Connection"
            print("Connected to database")
    
    def disconnect(self):
        if self.connection and self.transaction_depth == 0:
            print("Disconnected from database")
            self.connection = None
    
    @contextmanager
    def transaction(self):
        self.connect()
        self.transaction_depth += 1
        print(f"Begin transaction (depth: {self.transaction_depth})")
        
        try:
            yield self
            print(f"Commit transaction (depth: {self.transaction_depth})")
        except Exception as e:
            print(f"Rollback transaction (depth: {self.transaction_depth})")
            raise
        finally:
            self.transaction_depth -= 1
            if self.transaction_depth == 0:
                self.disconnect()

db = DatabaseConnection()

# Nested transactions
with db.transaction():
    print("Outer transaction work")
    with db.transaction():
        print("Inner transaction work")

# Resource pool with context awareness
class ResourcePool:
    def __init__(self, resource_factory, max_size=10):
        self.resource_factory = resource_factory
        self.max_size = max_size
        self.available = []
        self.in_use = set()
    
    @contextmanager
    def acquire(self):
        resource = self._get_resource()
        try:
            yield resource
        finally:
            self._release_resource(resource)
    
    def _get_resource(self):
        if self.available:
            resource = self.available.pop()
        elif len(self.in_use) < self.max_size:
            resource = self.resource_factory()
        else:
            raise Exception("No resources available")
        
        self.in_use.add(resource)
        return resource
    
    def _release_resource(self, resource):
        self.in_use.remove(resource)
        self.available.append(resource)

# Using ExitStack for dynamic context managers
@contextmanager
def managed_resources(*resources):
    with ExitStack() as stack:
        acquired = [stack.enter_context(resource) for resource in resources]
        yield acquired

# Example usage
# with managed_resources(open('file1.txt'), open('file2.txt')) as (f1, f2):
#     # Both files automatically closed
#     pass
```

---

## 29. Reflection and Introspection (`getattr`, `inspect`)

**Explanation:** Examining and modifying program structure at runtime, enabling dynamic behavior and metaprogramming.

**Use Cases:** Frameworks, ORMs, serialization, dynamic dispatch

**Code Example:**
```python
import inspect
from types import FunctionType

# Basic introspection
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hello, I'm {self.name}"
    
    def _private_method(self):
        return "Private"

person = Person("Alice", 30)

# Get attributes dynamically
name = getattr(person, 'name')
age = getattr(person, 'age', 0)  # Default value

# Set attributes dynamically
setattr(person, 'email', 'alice@example.com')

# Check attribute existence
has_name = hasattr(person, 'name')

# Get all attributes
attributes = dir(person)

# Filter methods
methods = [m for m in dir(person) if callable(getattr(person, m))]
public_methods = [m for m in methods if not m.startswith('_')]

# Inspect module
def example_function(x: int, y: str = "default") -> str:
    """Example function with type hints"""
    return f"{x}: {y}"

# Get function signature
sig = inspect.signature(example_function)
print(f"Parameters: {sig.parameters}")
print(f"Return annotation: {sig.return_annotation}")

# Get source code
source = inspect.getsource(example_function)

# Get function metadata
print(f"Name: {example_function.__name__}")
print(f"Doc: {example_function.__doc__}")
print(f"Module: {example_function.__module__}")

# Inspect call stack
def outer():
    def inner():
        frame = inspect.currentframe()
        print(f"Function: {frame.f_code.co_name}")
        print(f"Line: {frame.f_lineno}")
        
        # Get caller information
        caller_frame = frame.f_back
        print(f"Called from: {caller_frame.f_code.co_name}")
    inner()

# Dynamic class creation
def create_class(class_name, attributes):
    return type(class_name, (object,), attributes)

DynamicClass = create_class('DynamicClass', {
    'x': 10,
    'method': lambda self: self.x * 2
})

obj = DynamicClass()
print(obj.method())

# Introspection for serialization
def serialize_object(obj):
    if inspect.isclass(type(obj)):
        attributes = {}
        for name in dir(obj):
            if not name.startswith('_'):
                value = getattr(obj, name)
                if not callable(value):
                    attributes[name] = value
        return attributes
    return None

serialized = serialize_object(person)
print(serialized)

# Inspect class hierarchy
print(inspect.getmro(Person))  # Method Resolution Order

# Check if object is instance/subclass
print(isinstance(person, Person))
print(issubclass(Person, object))
```

---

## 30. Advanced Debugging and Profiling Tools

**Explanation:** Tools and techniques for identifying performance bottlenecks, memory leaks, and bugs in production code.

**Use Cases:** Performance optimization, debugging, production monitoring

**Code Example:**
```python
import cProfile
import pstats
import traceback
import sys
from io import StringIO
import time
import line_profiler
import memory_profiler

# Basic profiling with cProfile
def slow_function():
    total = 0
    for i in range(1000000):
        total += i ** 2
    return total

# Profile execution
profiler = cProfile.Profile()
profiler.enable()
result = slow_function()
profiler.disable()

# Print stats
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions

# Decorator for profiling
def profile_function(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats()
        return result
    return wrapper

@profile_function
def complex_calculation():
    return sum(i**2 for i in range(100000))

# Timing decorator
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def timed_function():
    time.sleep(1)
    return "Done"

# Custom exception handling with traceback
def debug_function():
    try:
        # Potentially buggy code
        x = 1 / 0
    except Exception as e:
        # Get detailed traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        # Format traceback
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_text = ''.join(tb_lines)
        print(tb_text)
        
        # Get stack trace
        stack = traceback.extract_tb(exc_traceback)
        for frame in stack:
            print(f"File: {frame.filename}, Line: {frame.lineno}, Function: {frame.name}")

# Memory profiling
# @memory_profiler.profile
def memory_intensive_function():
    large_list = [i for i in range(1000000)]
    return len(large_list)

# Line-by-line profiling
# @line_profiler.profile
def line_by_line_profiled():
    a = [i for i in range(1000)]
    b = [i**2 for i in a]
    c = sum(b)
    return c

# Debugging with pdb
import pdb

def buggy_function(x, y):
    result = x + y
    # pdb.set_trace()  # Breakpoint
    return result * 2

# Post-mortem debugging
def with_post_mortem():
    try:
        buggy_code = 1 / 0
    except:
        # pdb.post_mortem()  # Enter debugger at exception point
        pass

# Performance comparison
import timeit

# Compare different implementations
setup = "data = list(range(1000))"
list_comp = timeit.timeit('[x**2 for x in data]', setup=setup, number=10000)
map_func = timeit.timeit('list(map(lambda x: x**2, data))', setup=setup, number=10000)

print(f"List comprehension: {list_comp:.4f}s")
print(f"Map function: {map_func:.4f}s")

# Logging for debugging
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def debug_with_logging():
    logger.debug("Starting function")
    try:
        result = complex_calculation()
        logger.info(f"Result: {result}")
        return result
    except Exception as e:
        logger.exception("Error occurred")
        raise
```

---

## Summary

These 30 advanced Python topics cover the essential skills for professional Python development:

- **Performance**: Concurrency, parallelism, asyncio, GPU acceleration, vectorization
- **Code Quality**: Type hints, testing, logging, debugging, profiling
- **Design Patterns**: Decorators, context managers, design patterns, dependency injection
- **Advanced Features**: Metaclasses, descriptors, generators, coroutines
- **Production Ready**: Packaging, serialization, memory optimization, plugin architectures
- **Specialized Tools**: Web frameworks, data processing frameworks, profiling tools

Master these topics to write efficient, maintainable, and scalable Python applications!
