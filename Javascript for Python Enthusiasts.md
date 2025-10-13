# JavaScript for Python Developers: Complete Guide

## Table of Contents
1. [Basics & Syntax](#basics--syntax)
2. [Variables & Data Types](#variables--data-types)
3. [Operators](#operators)
4. [Control Flow](#control-flow)
5. [Functions](#functions)
6. [Data Structures](#data-structures)
7. [Object-Oriented Programming](#object-oriented-programming)
8. [Asynchronous Programming](#asynchronous-programming)
9. [Modules & Imports](#modules--imports)
10. [Error Handling](#error-handling)
11. [Modern JavaScript Features](#modern-javascript-features)

---

## Basics & Syntax

### Running Code

**Python:**
```python
# Run: python script.py
print("Hello World")
```

**JavaScript:**
```javascript
// Run: node script.js (Node.js) or in browser console
console.log("Hello World");
```

### Comments

**Python:**
```python
# Single line comment

"""
Multi-line comment
or docstring
"""
```

**JavaScript:**
```javascript
// Single line comment

/*
Multi-line comment
*/
```

### Semicolons
- **Python**: No semicolons needed
- **JavaScript**: Optional but recommended (automatic semicolon insertion can cause issues)

---

## Variables & Data Types

### Variable Declaration

**Python:**
```python
# Python has dynamic typing
x = 5
name = "Alice"
is_valid = True
```

**JavaScript:**
```javascript
// Three ways to declare variables
var x = 5;        // Old way (function-scoped, avoid)
let name = "Alice";  // Block-scoped, can reassign
const isValid = true; // Block-scoped, cannot reassign

// Use 'const' by default, 'let' when you need to reassign
```

### Data Types

**Python:**
```python
# Primitive types
integer = 42
floating = 3.14
string = "text"
boolean = True  # Capital T/F
none_value = None

# Check type
type(integer)  # <class 'int'>
```

**JavaScript:**
```javascript
// Primitive types
let integer = 42;
let floating = 3.14;  // No separate int/float, all are 'number'
let string = "text";
let boolean = true;  // lowercase
let undefinedValue = undefined;
let nullValue = null;

// Check type
typeof integer;  // "number"
```

### Strings

**Python:**
```python
# String operations
name = "Alice"
greeting = f"Hello, {name}"  # f-string
multi = """Multiple
lines"""
length = len(name)
upper = name.upper()
substring = name[0:3]
```

**JavaScript:**
```javascript
// String operations
let name = "Alice";
let greeting = `Hello, ${name}`;  // Template literal (backticks)
let multi = `Multiple
lines`;
let length = name.length;  // Property, not method
let upper = name.toUpperCase();
let substring = name.slice(0, 3);
```

### Type Conversion

**Python:**
```python
# Explicit conversion
str(42)        # "42"
int("42")      # 42
float("3.14")  # 3.14
bool(1)        # True
```

**JavaScript:**
```javascript
// Explicit conversion
String(42)        // "42"
Number("42")      // 42
parseInt("42")    // 42
parseFloat("3.14") // 3.14
Boolean(1)        // true

// Implicit conversion (be careful!)
"5" + 3   // "53" (string concatenation)
"5" - 3   // 2 (numeric subtraction)
```

---

## Operators

### Arithmetic Operators

**Python:**
```python
a + b    # Addition
a - b    # Subtraction
a * b    # Multiplication
a / b    # Division (always float)
a // b   # Floor division
a % b    # Modulus
a ** b   # Exponentiation
```

**JavaScript:**
```javascript
a + b    // Addition
a - b    // Subtraction
a * b    // Multiplication
a / b    // Division (may be float)
Math.floor(a / b)  // Floor division
a % b    // Modulus
a ** b   // Exponentiation (ES6+)
```

### Comparison Operators

**Python:**
```python
a == b   # Equal
a != b   # Not equal
a > b    # Greater than
a < b    # Less than
a >= b   # Greater or equal
a <= b   # Less or equal
```

**JavaScript:**
```javascript
// Loose equality (type coercion)
a == b   // Equal (avoid!)
a != b   // Not equal (avoid!)

// Strict equality (no type coercion) - PREFER THESE
a === b  // Strictly equal
a !== b  // Strictly not equal

a > b    // Greater than
a < b    // Less than
a >= b   // Greater or equal
a <= b   // Less or equal

// Example of difference
5 == "5"   // true (type coercion)
5 === "5"  // false (different types)
```

### Logical Operators

**Python:**
```python
a and b
a or b
not a
```

**JavaScript:**
```javascript
a && b   // AND
a || b   // OR
!a       // NOT

// Short-circuit evaluation works similarly
```

### Increment/Decrement

**Python:**
```python
x = 5
x += 1  # x = 6
x -= 1  # x = 5
```

**JavaScript:**
```javascript
let x = 5;
x++;     // x = 6 (post-increment)
++x;     // x = 7 (pre-increment)
x--;     // x = 6 (post-decrement)
x += 1;  // x = 7
```

---

## Control Flow

### If Statements

**Python:**
```python
if x > 10:
    print("Greater")
elif x > 5:
    print("Medium")
else:
    print("Small")
```

**JavaScript:**
```javascript
if (x > 10) {
    console.log("Greater");
} else if (x > 5) {
    console.log("Medium");
} else {
    console.log("Small");
}
```

### Ternary Operator

**Python:**
```python
result = "Yes" if x > 5 else "No"
```

**JavaScript:**
```javascript
let result = x > 5 ? "Yes" : "No";
```

### Switch Statement

**Python:**
```python
# Python 3.10+ match-case
match value:
    case 1:
        print("One")
    case 2:
        print("Two")
    case _:
        print("Other")
```

**JavaScript:**
```javascript
switch (value) {
    case 1:
        console.log("One");
        break;  // Important! Prevents fall-through
    case 2:
        console.log("Two");
        break;
    default:
        console.log("Other");
}
```

### For Loops

**Python:**
```python
# Range-based loop
for i in range(5):
    print(i)

# Iterate over list
for item in items:
    print(item)

# With index
for i, item in enumerate(items):
    print(i, item)
```

**JavaScript:**
```javascript
// Traditional for loop
for (let i = 0; i < 5; i++) {
    console.log(i);
}

// For-of loop (iterate values)
for (let item of items) {
    console.log(item);
}

// For-in loop (iterate keys/indices - avoid for arrays)
for (let key in object) {
    console.log(key);
}

// With index using entries()
for (let [i, item] of items.entries()) {
    console.log(i, item);
}
```

### While Loops

**Python:**
```python
while condition:
    # code
    break     # Exit loop
    continue  # Next iteration
```

**JavaScript:**
```javascript
while (condition) {
    // code
    break;     // Exit loop
    continue;  // Next iteration
}

// Do-while (runs at least once)
do {
    // code
} while (condition);
```

---

## Functions

### Basic Functions

**Python:**
```python
def greet(name):
    return f"Hello, {name}"

result = greet("Alice")
```

**JavaScript:**
```javascript
// Function declaration
function greet(name) {
    return `Hello, ${name}`;
}

let result = greet("Alice");
```

### Default Parameters

**Python:**
```python
def greet(name="Guest", age=0):
    return f"{name} is {age}"
```

**JavaScript:**
```javascript
function greet(name = "Guest", age = 0) {
    return `${name} is ${age}`;
}
```

### Variable Arguments

**Python:**
```python
def sum_all(*args):
    return sum(args)

def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")
```

**JavaScript:**
```javascript
// Rest parameters
function sumAll(...args) {
    return args.reduce((a, b) => a + b, 0);
}

// No direct kwargs equivalent, use object
function printInfo(options) {
    for (let [key, value] of Object.entries(options)) {
        console.log(`${key}: ${value}`);
    }
}
```

### Arrow Functions (Lambda)

**Python:**
```python
# Lambda
square = lambda x: x ** 2
add = lambda x, y: x + y
```

**JavaScript:**
```javascript
// Arrow functions
const square = (x) => x ** 2;
const add = (x, y) => x + y;

// Multiple statements need braces and explicit return
const complex = (x) => {
    let result = x ** 2;
    return result + 10;
};

// Single parameter doesn't need parentheses
const double = x => x * 2;
```

### Function Scope

**Python:**
```python
x = 10  # Global

def func():
    global x  # Modify global
    x = 20
    y = 5     # Local
```

**JavaScript:**
```javascript
let x = 10;  // Global (or module-scoped)

function func() {
    x = 20;      // Modifies outer x
    let y = 5;   // Local
    z = 15;      // Creates global (without let/const) - AVOID!
}
```

### Higher-Order Functions

**Python:**
```python
def apply_twice(func, x):
    return func(func(x))

result = apply_twice(lambda x: x * 2, 5)  # 20
```

**JavaScript:**
```javascript
function applyTwice(func, x) {
    return func(func(x));
}

let result = applyTwice(x => x * 2, 5);  // 20
```

---

## Data Structures

### Lists/Arrays

**Python:**
```python
# List
numbers = [1, 2, 3, 4, 5]
numbers.append(6)
numbers.insert(0, 0)
numbers.remove(3)
popped = numbers.pop()
length = len(numbers)
first = numbers[0]
last = numbers[-1]
sliced = numbers[1:3]
```

**JavaScript:**
```javascript
// Array
let numbers = [1, 2, 3, 4, 5];
numbers.push(6);           // Append
numbers.unshift(0);        // Insert at start
numbers.splice(numbers.indexOf(3), 1);  // Remove by value
let popped = numbers.pop();
let length = numbers.length;  // Property
let first = numbers[0];
let last = numbers[numbers.length - 1];  // No negative indexing
let sliced = numbers.slice(1, 3);  // End is exclusive
```

### List/Array Methods

**Python:**
```python
numbers = [1, 2, 3, 4, 5]

# Map
doubled = list(map(lambda x: x * 2, numbers))
# or
doubled = [x * 2 for x in numbers]

# Filter
evens = list(filter(lambda x: x % 2 == 0, numbers))
# or
evens = [x for x in numbers if x % 2 == 0]

# Reduce
from functools import reduce
total = reduce(lambda a, b: a + b, numbers)

# Find
found = next((x for x in numbers if x > 3), None)

# Check existence
exists = 3 in numbers
```

**JavaScript:**
```javascript
let numbers = [1, 2, 3, 4, 5];

// Map
let doubled = numbers.map(x => x * 2);

// Filter
let evens = numbers.filter(x => x % 2 === 0);

// Reduce
let total = numbers.reduce((a, b) => a + b, 0);

// Find
let found = numbers.find(x => x > 3);  // Returns value or undefined

// Find index
let index = numbers.findIndex(x => x > 3);

// Check existence
let exists = numbers.includes(3);

// Every/Some
let allPositive = numbers.every(x => x > 0);
let hasEven = numbers.some(x => x % 2 === 0);

// ForEach (no return value)
numbers.forEach((x, i) => console.log(i, x));
```

### Dictionaries/Objects

**Python:**
```python
# Dictionary
person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}

person["name"]           # Access
person["email"] = "..."  # Add
del person["city"]       # Delete
"name" in person         # Check key
person.keys()            # Keys
person.values()          # Values
person.items()           # Key-value pairs

# Iterate
for key, value in person.items():
    print(key, value)
```

**JavaScript:**
```javascript
// Object (like dict, but keys are always strings)
let person = {
    name: "Alice",      // Key without quotes (becomes string)
    age: 30,
    city: "NYC"
};

person.name;             // Access (dot notation)
person["name"];          // Access (bracket notation)
person.email = "...";    // Add
delete person.city;      // Delete
"name" in person;        // Check key
Object.keys(person);     // Keys array
Object.values(person);   // Values array
Object.entries(person);  // [[key, value], ...] array

// Iterate
for (let [key, value] of Object.entries(person)) {
    console.log(key, value);
}
```

### Sets

**Python:**
```python
# Set
numbers = {1, 2, 3, 4, 5}
numbers.add(6)
numbers.remove(3)
numbers.discard(10)  # No error if not exists
exists = 3 in numbers

# Set operations
a = {1, 2, 3}
b = {3, 4, 5}
union = a | b
intersection = a & b
difference = a - b
```

**JavaScript:**
```javascript
// Set
let numbers = new Set([1, 2, 3, 4, 5]);
numbers.add(6);
numbers.delete(3);
numbers.has(3);          // Check existence
numbers.size;            // Length (property)

// Set operations (manual)
let a = new Set([1, 2, 3]);
let b = new Set([3, 4, 5]);

// Union
let union = new Set([...a, ...b]);

// Intersection
let intersection = new Set([...a].filter(x => b.has(x)));

// Difference
let difference = new Set([...a].filter(x => !b.has(x)));

// Iterate
for (let value of numbers) {
    console.log(value);
}
```

### Maps

**Python:**
```python
# Dict works for most cases
# For non-string keys, use dict directly
data = {
    1: "one",
    (1, 2): "tuple key"
}
```

**JavaScript:**
```javascript
// Map (for any type of keys, including objects)
let map = new Map();
map.set("key", "value");
map.set(1, "one");
map.set({id: 1}, "object key");

map.get("key");          // Get value
map.has("key");          // Check existence
map.delete("key");       // Delete
map.size;                // Length
map.clear();             // Remove all

// Iterate
for (let [key, value] of map) {
    console.log(key, value);
}

// Initialize with array
let map2 = new Map([
    ["key1", "value1"],
    ["key2", "value2"]
]);
```

### Tuples (Immutability)

**Python:**
```python
# Tuple (immutable)
point = (10, 20)
x, y = point  # Destructuring
```

**JavaScript:**
```javascript
// No built-in tuple, use array
// For immutability, use const (but array contents can still change)
const point = [10, 20];
const [x, y] = point;  // Destructuring

// For true immutability
const immutablePoint = Object.freeze([10, 20]);
```

### Destructuring

**Python:**
```python
# List
a, b, c = [1, 2, 3]
a, *rest = [1, 2, 3, 4]  # rest = [2, 3, 4]

# Dict
data = {"name": "Alice", "age": 30}
name = data["name"]
```

**JavaScript:**
```javascript
// Array destructuring
let [a, b, c] = [1, 2, 3];
let [first, ...rest] = [1, 2, 3, 4];  // rest = [2, 3, 4]

// Object destructuring
let person = {name: "Alice", age: 30};
let {name, age} = person;
let {name: fullName} = person;  // Rename variable

// Default values
let {name, city = "Unknown"} = person;

// Nested destructuring
let {address: {street}} = {address: {street: "Main St"}};
```

### Spread/Rest Operator

**Python:**
```python
# Spread
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = [*list1, *list2]

dict1 = {"a": 1}
dict2 = {"b": 2}
merged = {**dict1, **dict2}
```

**JavaScript:**
```javascript
// Spread
let list1 = [1, 2, 3];
let list2 = [4, 5, 6];
let combined = [...list1, ...list2];

let obj1 = {a: 1};
let obj2 = {b: 2};
let merged = {...obj1, ...obj2};

// Also used in function calls
function sum(a, b, c) {
    return a + b + c;
}
let numbers = [1, 2, 3];
sum(...numbers);  // Spreads array as arguments
```

---

## Object-Oriented Programming

### Classes

**Python:**
```python
class Person:
    # Class variable
    species = "Homo sapiens"
    
    def __init__(self, name, age):
        # Instance variables
        self.name = name
        self.age = age
        self._protected = "protected"
        self.__private = "private"
    
    def greet(self):
        return f"Hello, I'm {self.name}"
    
    @staticmethod
    def static_method():
        return "Static"
    
    @classmethod
    def class_method(cls):
        return cls.species

# Create instance
person = Person("Alice", 30)
print(person.greet())
```

**JavaScript:**
```javascript
class Person {
    // Class field (ES2022+)
    static species = "Homo sapiens";
    
    constructor(name, age) {
        // Instance properties
        this.name = name;
        this.age = age;
        this._protected = "protected";  // Convention only
        // Private fields use #
        this.#private = "private";
    }
    
    // Private field declaration
    #private;
    
    greet() {
        return `Hello, I'm ${this.name}`;
    }
    
    static staticMethod() {
        return "Static";
    }
}

// Create instance
let person = new Person("Alice", 30);
console.log(person.greet());
```

### Inheritance

**Python:**
```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "Some sound"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)
        self.breed = breed
    
    def speak(self):
        return "Woof!"

dog = Dog("Buddy", "Golden Retriever")
print(dog.speak())  # "Woof!"
```

**JavaScript:**
```javascript
class Animal {
    constructor(name) {
        this.name = name;
    }
    
    speak() {
        return "Some sound";
    }
}

class Dog extends Animal {
    constructor(name, breed) {
        super(name);  // Must call super first
        this.breed = breed;
    }
    
    speak() {
        return "Woof!";
    }
}

let dog = new Dog("Buddy", "Golden Retriever");
console.log(dog.speak());  // "Woof!"
```

### Getters and Setters

**Python:**
```python
class Person:
    def __init__(self, name):
        self._name = name
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        if len(value) > 0:
            self._name = value

person = Person("Alice")
print(person.name)  # Getter
person.name = "Bob"  # Setter
```

**JavaScript:**
```javascript
class Person {
    constructor(name) {
        this._name = name;
    }
    
    get name() {
        return this._name;
    }
    
    set name(value) {
        if (value.length > 0) {
            this._name = value;
        }
    }
}

let person = new Person("Alice");
console.log(person.name);  // Getter
person.name = "Bob";        // Setter
```

---

## Asynchronous Programming

### Callbacks (Old Style)

**Python:**
```python
# Python doesn't use callbacks as primary pattern
def fetch_data(callback):
    # Simulated async operation
    data = "result"
    callback(data)

fetch_data(lambda data: print(data))
```

**JavaScript:**
```javascript
// Callback pattern (older, callback hell)
function fetchData(callback) {
    setTimeout(() => {
        callback("result");
    }, 1000);
}

fetchData((data) => {
    console.log(data);
});
```

### Promises

**Python:**
```python
# Python uses asyncio, not promises
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return "result"

# Run async function
asyncio.run(fetch_data())
```

**JavaScript:**
```javascript
// Promise
function fetchData() {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve("result");
            // or reject(new Error("Failed"));
        }, 1000);
    });
}

// Using promise
fetchData()
    .then(data => console.log(data))
    .catch(error => console.error(error))
    .finally(() => console.log("Done"));

// Chaining
fetchData()
    .then(data => data.toUpperCase())
    .then(upper => console.log(upper));
```

### Async/Await

**Python:**
```python
import asyncio

async def fetch_user():
    await asyncio.sleep(1)
    return {"name": "Alice"}

async def fetch_posts():
    await asyncio.sleep(1)
    return ["post1", "post2"]

async def main():
    # Sequential
    user = await fetch_user()
    posts = await fetch_posts()
    
    # Parallel
    user, posts = await asyncio.gather(
        fetch_user(),
        fetch_posts()
    )

asyncio.run(main())
```

**JavaScript:**
```javascript
async function fetchUser() {
    await new Promise(resolve => setTimeout(resolve, 1000));
    return {name: "Alice"};
}

async function fetchPosts() {
    await new Promise(resolve => setTimeout(resolve, 1000));
    return ["post1", "post2"];
}

async function main() {
    // Sequential
    let user = await fetchUser();
    let posts = await fetchPosts();
    
    // Parallel
    let [user2, posts2] = await Promise.all([
        fetchUser(),
        fetchPosts()
    ]);
}

// Can call async functions
main().catch(console.error);
```

### Promise Utilities

**Python:**
```python
# asyncio.gather - all tasks
results = await asyncio.gather(task1(), task2())

# asyncio.wait with return_when
done, pending = await asyncio.wait(
    [task1(), task2()],
    return_when=asyncio.FIRST_COMPLETED
)
```

**JavaScript:**
```javascript
// Promise.all - wait for all (fails if any fails)
let results = await Promise.all([promise1, promise2]);

// Promise.allSettled - wait for all (never fails)
let results = await Promise.allSettled([promise1, promise2]);
// [{status: "fulfilled", value: ...}, {status: "rejected", reason: ...}]

// Promise.race - first to complete
let first = await Promise.race([promise1, promise2]);

// Promise.any - first to succeed
let first = await Promise.any([promise1, promise2]);
```

---

## Modules & Imports

### Python Modules

**Python:**
```python
# math_utils.py
def add(a, b):
    return a + b

PI = 3.14159

# main.py
import math_utils
print(math_utils.add(1, 2))

from math_utils import add, PI
print(add(1, 2))

from math_utils import add as addition
print(addition(1, 2))

from math_utils import *  # Import all (not recommended)
```

### JavaScript Modules (ES6)

**JavaScript:**
```javascript
// mathUtils.js
export function add(a, b) {
    return a + b;
}

export const PI = 3.14159;

// Default export (one per file)
export default function multiply(a, b) {
    return a * b;
}

// main.js
import multiply from './mathUtils.js';  // Default import
import { add, PI } from './mathUtils.js';  // Named imports
import { add as addition } from './mathUtils.js';  // Rename
import * as MathUtils from './mathUtils.js';  // Import all

console.log(add(1, 2));
console.log(MathUtils.PI);
```

### CommonJS (Node.js)

**JavaScript:**
```javascript
// mathUtils.js
function add(a, b) {
    return a + b;
}

module.exports = { add, PI: 3.14159 };
// or
exports.add = add;

// main.js
const { add, PI } = require('./mathUtils');
const MathUtils = require('./mathUtils');

console.log(add(1, 2));
```

---

## Error Handling

### Try/Catch

**Python:**
```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"General error: {e}")
else:
    print("No error occurred")
finally:
    print("Always executed")
```

**JavaScript:**
```javascript
try {
    let result = 10 / 0;  // No error in JS, returns Infinity
    throw new Error("Custom error");
} catch (error) {
    console.log(`Error: ${error.message}`);
    // No specific exception types by default
} finally {
    console.log("Always executed");
}
```

### Custom Errors

**Python:**
```python
class CustomError(Exception):
    pass

try:
    raise CustomError("Something went wrong")
except CustomError as e:
    print(e)
```

**JavaScript:**
```javascript
class CustomError extends Error {
    constructor(message) {
        super(message);
        this.name = "CustomError";
    }
}

try {
    throw new CustomError("Something went wrong");
} catch (error) {
    if (error instanceof CustomError) {
        console.log(error.message);
    }
}
```

### Async Error Handling

**Python:**
```python
async def fetch_data():
    try:
        result = await some_async_operation()
    except Exception as e:
        print(f"Error: {e}")
```

**JavaScript:**
```javascript
async function fetchData() {
    try {
        let result = await someAsyncOperation();
    } catch (error) {
        console.log(`Error: ${error.message}`);
    }
}

// Or with promises
someAsyncOperation()
    .catch(error => console.log(error));
```

---

## Modern JavaScript Features

### Optional Chaining

**Python:**
```python
# No direct equivalent, need manual checks
user = {"profile": {"name": "Alice"}}
name = user.get("profile", {}).get("name")
```

**JavaScript:**
```javascript
let user = {profile: {name: "Alice"}};
let name = user?.profile?.name;  // "Alice"
let missing = user?.address?.street;  // undefined (no error)

// With functions
obj?.method?.();

// With arrays
arr?.[0];
```

### Nullish Coalescing

**Python:**
```python
# Use 'or' but beware of falsy values
value = user_input or "default"

# More explicit
value = user_input if user_input is not None else "default"
```

**JavaScript:**
```javascript
// || operator (falsy values)
let value = userInput || "default";  // 0, "", false also trigger default

// ?? operator (only null/undefined)
let value = userInput ?? "default";  // Only null/undefined trigger default

let count = 0;
console.log(count || 10);   // 10
console.log(count ?? 10);   // 0
```

### Template Literals

**Python:**
```python
name = "Alice"
age = 30
message = f"Hello, {name}. You are {age} years old."

# Multi-line
message = f"""
Hello, {name}.
You are {age} years old.
"""
```

**JavaScript:**
```javascript
let name = "Alice";
let age = 30;
let message = `Hello, ${name}. You are ${age} years old.`;

// Multi-line
let message = `
Hello, ${name}.
You are ${age} years old.
`;

// Expression evaluation
let result = `2 + 2 = ${2 + 2}`;
```

### Tagged Templates

**Python:**
```python
# No direct equivalent
```

**JavaScript:**
```javascript
// Advanced feature - template literals as function arguments
function highlight(strings, ...values) {
    return strings.reduce((result, str, i) => {
        return result + str + (values[i] ? `<mark>${values[i]}</mark>` : '');
    }, '');
}

let name = "Alice";
let html = highlight`Hello, ${name}!`;  // "Hello, <mark>Alice</mark>!"
```

### Generators

**Python:**
```python
def count_up_to(n):
    i = 0
    while i < n:
        yield i
        i += 1

for num in count_up_to(5):
    print(num)

# Generator expression
squares = (x**2 for x in range(10))
```

**JavaScript:**
```javascript
function* countUpTo(n) {
    let i = 0;
    while (i < n) {
        yield i;
        i++;
    }
}

for (let num of countUpTo(5)) {
    console.log(num);
}

// Manual iteration
let gen = countUpTo(3);
console.log(gen.next());  // {value: 0, done: false}
console.log(gen.next());  // {value: 1, done: false}
console.log(gen.next());  // {value: 2, done: false}
console.log(gen.next());  // {value: undefined, done: true}
```

### Symbols

**Python:**
```python
# No direct equivalent
# Python uses string keys
```

**JavaScript:**
```javascript
// Symbols create unique identifiers
let id = Symbol("id");
let id2 = Symbol("id");
console.log(id === id2);  // false (each symbol is unique)

// Use as object keys (won't conflict)
let obj = {
    [id]: "value"
};

// Well-known symbols
class MyArray {
    [Symbol.iterator]() {
        // Custom iteration behavior
    }
}
```

### Proxy and Reflect

**Python:**
```python
# Python uses __getattr__, __setattr__, etc.
class Proxy:
    def __init__(self, target):
        self._target = target
    
    def __getattr__(self, name):
        print(f"Getting {name}")
        return getattr(self._target, name)
```

**JavaScript:**
```javascript
// Proxy for meta-programming
let target = {name: "Alice"};

let proxy = new Proxy(target, {
    get(target, property) {
        console.log(`Getting ${property}`);
        return target[property];
    },
    set(target, property, value) {
        console.log(`Setting ${property} to ${value}`);
        target[property] = value;
        return true;
    }
});

proxy.name;  // Logs: "Getting name", returns "Alice"
proxy.age = 30;  // Logs: "Setting age to 30"
```

### WeakMap and WeakSet

**Python:**
```python
import weakref

# WeakValueDictionary
cache = weakref.WeakValueDictionary()
```

**JavaScript:**
```javascript
// WeakMap - keys are objects, allows garbage collection
let weakMap = new WeakMap();
let obj = {id: 1};
weakMap.set(obj, "metadata");
console.log(weakMap.get(obj));  // "metadata"

// When obj is garbage collected, entry is automatically removed

// WeakSet - similar for sets
let weakSet = new WeakSet();
weakSet.add(obj);
console.log(weakSet.has(obj));  // true
```

---

## JavaScript Specific Concepts

### This Keyword

**Python:**
```python
class MyClass:
    def __init__(self):
        self.value = 10
    
    def method(self):
        return self.value  # self is explicit
```

**JavaScript:**
```javascript
// 'this' context can change based on how function is called
let obj = {
    value: 10,
    method: function() {
        return this.value;  // 'this' refers to obj
    },
    arrowMethod: () => {
        return this.value;  // 'this' from outer scope (NOT obj)
    }
};

console.log(obj.method());  // 10

// Problem: losing context
let method = obj.method;
console.log(method());  // undefined (this is lost)

// Solutions:
// 1. Bind
let boundMethod = obj.method.bind(obj);
console.log(boundMethod());  // 10

// 2. Arrow function (preserves context)
let arrowFn = () => obj.method();

// 3. Call/Apply
obj.method.call(obj);
obj.method.apply(obj, []);
```

### Closures

**Python:**
```python
def outer(x):
    def inner(y):
        return x + y  # Accesses outer scope
    return inner

add_five = outer(5)
print(add_five(3))  # 8
```

**JavaScript:**
```javascript
function outer(x) {
    return function inner(y) {
        return x + y;  // Accesses outer scope
    };
}

let addFive = outer(5);
console.log(addFive(3));  // 8

// Common use: private variables
function createCounter() {
    let count = 0;  // Private
    return {
        increment: () => ++count,
        decrement: () => --count,
        getCount: () => count
    };
}

let counter = createCounter();
console.log(counter.increment());  // 1
console.log(counter.getCount());   // 1
// count is not directly accessible
```

### Immediately Invoked Function Expression (IIFE)

**Python:**
```python
# No direct equivalent, but can do:
result = (lambda: "value")()
```

**JavaScript:**
```javascript
// IIFE - function that runs immediately
(function() {
    let privateVar = "secret";
    console.log("Running immediately");
})();

// Arrow function IIFE
(() => {
    console.log("Also runs immediately");
})();

// Use case: create private scope
let module = (function() {
    let privateVar = 0;
    return {
        getVar: () => privateVar,
        increment: () => ++privateVar
    };
})();
```

### Event Loop and Execution

**Python:**
```python
# Python is synchronous by default
print("First")
print("Second")
print("Third")
```

**JavaScript:**
```javascript
// JavaScript is single-threaded with event loop
console.log("First");

setTimeout(() => {
    console.log("Second");
}, 0);  // Even with 0ms, runs after current execution

console.log("Third");

// Output: First, Third, Second

// Microtasks (Promises) vs Macrotasks (setTimeout)
console.log("1");

setTimeout(() => console.log("2"), 0);

Promise.resolve().then(() => console.log("3"));

console.log("4");

// Output: 1, 4, 3, 2
// Promises run before setTimeout
```

### Prototypes

**Python:**
```python
# Python uses classes, not prototypes
class Animal:
    def speak(self):
        return "sound"

# Inheritance is class-based
```

**JavaScript:**
```javascript
// Every object has a prototype
function Animal(name) {
    this.name = name;
}

// Add method to prototype (shared by all instances)
Animal.prototype.speak = function() {
    return "sound";
};

let dog = new Animal("Buddy");
console.log(dog.speak());  // "sound"

// Prototype chain
console.log(dog.__proto__ === Animal.prototype);  // true
console.log(Animal.prototype.__proto__ === Object.prototype);  // true

// Modern: use classes (syntactic sugar over prototypes)
class ModernAnimal {
    constructor(name) {
        this.name = name;
    }
    
    speak() {
        return "sound";
    }
}
```

### JSON

**Python:**
```python
import json

# Object to JSON
data = {"name": "Alice", "age": 30}
json_string = json.dumps(data)

# JSON to object
parsed = json.loads(json_string)

# File operations
with open("data.json", "w") as f:
    json.dump(data, f)

with open("data.json", "r") as f:
    loaded = json.load(f)
```

**JavaScript:**
```javascript
// Object to JSON
let data = {name: "Alice", age: 30};
let jsonString = JSON.stringify(data);

// Pretty print
let prettyJson = JSON.stringify(data, null, 2);

// JSON to object
let parsed = JSON.parse(jsonString);

// Custom serialization
let custom = JSON.stringify(data, (key, value) => {
    if (key === "age") return undefined;  // Exclude
    return value;
});

// Note: JSON doesn't support functions, undefined, symbols
let obj = {
    name: "Alice",
    fn: () => {},      // Will be excluded
    value: undefined   // Will be excluded
};
```

### Regular Expressions

**Python:**
```python
import re

pattern = r"\d+"
text = "I have 123 apples"

# Match
match = re.search(pattern, text)
if match:
    print(match.group())  # "123"

# Find all
matches = re.findall(r"\d+", text)

# Replace
result = re.sub(r"\d+", "many", text)

# Split
parts = re.split(r"\s+", text)
```

**JavaScript:**
```javascript
let pattern = /\d+/;
let text = "I have 123 apples";

// Test
console.log(pattern.test(text));  // true

// Match
let match = text.match(pattern);
console.log(match[0]);  // "123"

// Find all (with 'g' flag)
let matches = text.match(/\d+/g);  // ["123"]

// Replace
let result = text.replace(/\d+/, "many");
let resultAll = text.replace(/\d+/g, "many");  // Replace all

// Split
let parts = text.split(/\s+/);

// Flags
let regex = /pattern/gi;  // g=global, i=case-insensitive
// Other flags: m=multiline, s=dotAll, u=unicode, y=sticky

// Groups
let datePattern = /(\d{4})-(\d{2})-(\d{2})/;
let dateMatch = "2024-03-15".match(datePattern);
console.log(dateMatch[1]);  // "2024"
console.log(dateMatch[2]);  // "03"
```

---

## Common Patterns and Idioms

### Checking if Variable is Defined

**Python:**
```python
# Check if variable exists
try:
    variable
except NameError:
    print("Not defined")

# Check if None
if variable is None:
    print("Is None")

# Check if falsy
if not variable:
    print("Falsy")
```

**JavaScript:**
```javascript
// Check if undefined
if (typeof variable === 'undefined') {
    console.log("Not defined");
}

// Check if null or undefined
if (variable == null) {  // Loose equality
    console.log("Is null or undefined");
}

if (variable === null) {
    console.log("Is null");
}

if (variable === undefined) {
    console.log("Is undefined");
}

// Check if falsy (0, "", false, null, undefined, NaN)
if (!variable) {
    console.log("Falsy");
}

// Check if value exists (not null/undefined)
if (variable != null) {
    console.log("Has value");
}
```

### Checking Types

**Python:**
```python
isinstance(x, int)
isinstance(x, (int, float))
type(x) == int
```

**JavaScript:**
```javascript
typeof x === 'number'
typeof x === 'string'
typeof x === 'boolean'
typeof x === 'function'
typeof x === 'object'  // Arrays, objects, null!
typeof x === 'undefined'

// Better checks
Array.isArray(x)
x === null
x instanceof Date
x instanceof MyClass

// Checking for NaN
Number.isNaN(x)  // Prefer this
isNaN(x)         // Less reliable
```

### Iterating Objects

**Python:**
```python
person = {"name": "Alice", "age": 30}

# Keys
for key in person:
    print(key)

# Values
for value in person.values():
    print(value)

# Items
for key, value in person.items():
    print(key, value)
```

**JavaScript:**
```javascript
let person = {name: "Alice", age: 30};

// Keys
for (let key in person) {
    if (person.hasOwnProperty(key)) {  // Check own properties
        console.log(key);
    }
}

// Modern way - keys
for (let key of Object.keys(person)) {
    console.log(key);
}

// Values
for (let value of Object.values(person)) {
    console.log(value);
}

// Entries
for (let [key, value] of Object.entries(person)) {
    console.log(key, value);
}
```

### Cloning Objects/Arrays

**Python:**
```python
import copy

# Shallow copy
list_copy = original_list.copy()
dict_copy = original_dict.copy()

# Deep copy
deep_copy = copy.deepcopy(original)
```

**JavaScript:**
```javascript
// Shallow copy - arrays
let copy = [...original];
let copy2 = original.slice();
let copy3 = Array.from(original);

// Shallow copy - objects
let copy = {...original};
let copy2 = Object.assign({}, original);

// Deep copy (simple objects only, no functions/dates)
let deepCopy = JSON.parse(JSON.stringify(original));

// Deep copy (modern, with functions/dates)
let deepCopy2 = structuredClone(original);
```

### Default Values

**Python:**
```python
# Dict with default
value = my_dict.get("key", "default")

# List access with default
value = my_list[0] if len(my_list) > 0 else "default"
```

**JavaScript:**
```javascript
// Object with default
let value = obj.key || "default";  // Falsy check
let value2 = obj.key ?? "default";  // Nullish check

// Array access with default
let value = arr[0] || "default";

// Optional chaining with default
let value = obj?.nested?.prop ?? "default";

// Destructuring with default
let {name = "Unknown"} = obj;
let [first = "default"] = arr;
```

---

## Browser-Specific Features

### DOM Manipulation

**Python:**
```python
# Python doesn't run in browser
# Use frameworks like Pyodide for browser Python
```

**JavaScript:**
```javascript
// Select elements
let element = document.getElementById("myId");
let elements = document.getElementsByClassName("myClass");
let elements2 = document.getElementsByTagName("div");
let element2 = document.querySelector(".myClass");
let elements3 = document.querySelectorAll(".myClass");

// Modify content
element.textContent = "New text";
element.innerHTML = "<strong>HTML</strong>";

// Modify attributes
element.setAttribute("data-value", "123");
element.id = "newId";
element.className = "newClass";
element.classList.add("another-class");
element.classList.remove("old-class");
element.classList.toggle("active");

// Modify styles
element.style.color = "red";
element.style.backgroundColor = "blue";

// Create and append elements
let newDiv = document.createElement("div");
newDiv.textContent = "Hello";
document.body.appendChild(newDiv);

// Remove element
element.remove();
```

### Event Handling

**JavaScript:**
```javascript
// Add event listener
element.addEventListener("click", function(event) {
    console.log("Clicked!", event);
});

// Arrow function
element.addEventListener("click", (e) => {
    console.log(e.target);  // Element that was clicked
});

// Remove event listener
function handleClick(e) {
    console.log("Clicked");
}
element.addEventListener("click", handleClick);
element.removeEventListener("click", handleClick);

// Event delegation
document.body.addEventListener("click", (e) => {
    if (e.target.matches(".button")) {
        console.log("Button clicked");
    }
});

// Prevent default behavior
form.addEventListener("submit", (e) => {
    e.preventDefault();
    console.log("Form not submitted");
});
```

### Local Storage

**JavaScript:**
```javascript
// Store data (strings only)
localStorage.setItem("key", "value");
localStorage.setItem("user", JSON.stringify({name: "Alice"}));

// Retrieve data
let value = localStorage.getItem("key");
let user = JSON.parse(localStorage.getItem("user"));

// Remove data
localStorage.removeItem("key");

// Clear all
localStorage.clear();

// sessionStorage (cleared when tab closes)
sessionStorage.setItem("key", "value");
```

### Fetch API

**JavaScript:**
```javascript
// GET request
fetch("https://api.example.com/data")
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error(error));

// With async/await
async function fetchData() {
    try {
        let response = await fetch("https://api.example.com/data");
        let data = await response.json();
        console.log(data);
    } catch (error) {
        console.error(error);
    }
}

// POST request
fetch("https://api.example.com/data", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({name: "Alice"})
})
    .then(response => response.json())
    .then(data => console.log(data));
```

---

## Key Differences Summary

### Python vs JavaScript Quick Reference

| Feature | Python | JavaScript |
|---------|--------|------------|
| **Syntax** | Indentation-based | Braces `{}` |
| **Variables** | Dynamic, no declaration | `let`, `const`, `var` |
| **Equality** | `==`, `is` | `===` (strict), `==` (loose) |
| **Boolean** | `True`, `False` | `true`, `false` |
| **Null** | `None` | `null`, `undefined` |
| **Arrays** | Lists `[]` | Arrays `[]` |
| **Dicts** | `dict` | Objects `{}`, `Map` |
| **Functions** | `def` keyword | `function`, arrow `=>` |
| **String format** | f-strings | Template literals `` ` `` |
| **This/Self** | Explicit `self` | Implicit `this` |
| **Async** | `async`/`await` | Promises, `async`/`await` |
| **Classes** | Class-based | Prototype-based (class syntax) |
| **Modules** | `import` | `import`/`export`, `require` |
| **Scope** | Function, global | Function, block, global |
| **Typing** | Dynamic | Dynamic (TypeScript for static) |

### Common Gotchas for Python Developers

1. **Semicolons**: Optional but can cause issues with automatic insertion
2. **`==` vs `===`**: Always use `===` for strict equality
3. **`this` context**: Can change unexpectedly, use arrow functions or bind
4. **Truthy/Falsy**: More values are falsy (0, "", NaN, null, undefined)
5. **Array/Object comparison**: Compare by reference, not value
6. **Hoisting**: Variables and functions are hoisted to top of scope
7. **Block scope**: `var` is function-scoped, `let`/`const` are block-scoped
8. **No negative indexing**: `arr[-1]` doesn't work like Python
9. **Type coercion**: Implicit conversions can be surprising
10. **Async is everywhere**: Many operations are asynchronous in JavaScript

---

## Best Practices

### Code Style

```javascript
// Use const by default, let when reassigning
const API_KEY = "abc123";
let counter = 0;

// Use meaningful names
const getUserById = (id) => { /* */ };

// Use === not ==
if (value === 42) { /* */ }

// Use template literals
const message = `Hello, ${name}!`;

// Use arrow functions for callbacks
array.map(item => item * 2);

// Use destructuring
const {name, age} = user;
const [first, second] = array;

// Use spread operator
const newArray = [...oldArray, newItem];
const newObject = {...oldObject, newProp: value};

// Use optional chaining
const street = user?.address?.street;

// Use default parameters
function greet(name = "Guest") { /* */ }
```

### Modern JavaScript

```javascript
// Prefer async/await over promises
async function fetchUser() {
    const response = await fetch("/api/user");
    return response.json();
}

// Use Array methods instead of loops
const doubled = numbers.map(n => n * 2);
const evens = numbers.filter(n => n % 2 === 0);
const sum = numbers.reduce((a, b) => a + b, 0);

// Use object/array shorthand
const name = "Alice";
const age = 30;
const user = {name, age};  // Same as {name: name, age: age}

// Use modules
import {helper} from './utils.js';
export const myFunction = () => { /* */ };
```

---

## Next Steps

1. **Practice**: Build small projects to understand async patterns
2. **Learn TypeScript**: Adds static typing to JavaScript
3. **Explore Frameworks**: React, Vue, Angular for front-end
4. **Node.js**: JavaScript on the server-side
5. **Testing**: Jest, Mocha for unit testing
6. **Build Tools**: Webpack, Vite, Babel for modern development

---

This guide covers the essentials you need to transition from Python to JavaScript. The key is to practice and gradually build familiarity with JavaScript's asynchronous nature and prototype-based OOP model!
