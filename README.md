# epypen  
_A Lightweight Translation Decorator for Python Objects_

[![PyPI version](https://img.shields.io/pypi/v/epypen.svg)](https://pypi.org/project/epypen/)  
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://choosealicense.com/licenses/gpl-3.0/)  
[![Build Status](https://img.shields.io/github/workflow/status/avivgood/epypen/CI)](https://github.com/avivgood/epypen/actions)  
[![Coverage Status](https://img.shields.io/codecov/c/github/avivgood/epypen.svg)](https://codecov.io/gh/avivgood/epypen)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Conversion Example](#basic-conversion-example)
  - [Main Usage: Converting to Pydantic Models](#main-usage-converting-to-pydantic-models)
- [Advanced Usage](#advanced-usage)
  - [Understanding `to()` and `via()`](#understanding-to-and-via)
  - [Customizing Conversions](#customizing-conversions)
  - [Handling Conversion Errors](#handling-conversion-errors)
- [Gotchas](#gotchas)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

**epypen** is a lightweight translation decorator that bridges the gap between raw Python objects and type-annotated data. It allows automatic conversion of function arguments and return values based on type annotations. This is especially useful in projects that use Pydantic models (for example, FastAPI applications) to perform validation and transformation without writing extra boilerplate code.

---

## Features

- **Automatic Parameter Conversion:** Converts function arguments to their expected types based on annotations.
- **Return Type Translation:** Adjusts function outputs to match annotated return types.
- **Signature Preservation:** Updates function signatures according to annotations—e.g., using `via()` on a parameter will modify its annotation in the final signature.
- **Supports Multiple Function Types:** Works with synchronous, asynchronous, instance, class, and static methods.
- **Custom Conversion Logic:** Users can supply custom conversion functions via `parameter_conversions` and `return_value_conversions`.
- **Built on Modern Python Tools:** Leverages Pydantic, typing-extensions, and Python’s introspection capabilities.

---

## Installation

Install **epypen** directly from PyPI:

```bash
pip install epypen
```

---

## Usage

### Basic Conversion Example

This simple example demonstrates how **epypen** converts parameter types and the return value based on annotations:

```python
from epypen import converted

@converted
def _sum(a: int, b: int) -> int:
    return a + b

# Although passed as strings, the arguments are converted to integers.
assert _sum("2", "3") == 5
```

### Main Usage: Converting to Pydantic Models

A common use case for **epypen** is to seamlessly convert dictionaries into Pydantic models. For example:

```python
from pydantic import BaseModel
from epypen import converted, to

# Define Pydantic models for input and output.
class X(BaseModel):
    x: int

class Y(BaseModel):
    y: int

class Result(BaseModel):
    z: int

@converted
def add(x: X, y: Y) -> Result:
    return Result(z=x.x + y.y)

# You can pass dictionaries and they will be automatically converted:
result = add({"x": "2"}, {"y": 3})
assert result == Result(z=5)
```

---

## Advanced Usage

### Understanding `to()` and `via()`

- **`to()`**:  
  This helper is used in return type annotations to specify the *target* type. For example, if your function returns a string that should be converted to an integer, wrap the return annotation with `to(int)`.

  ```python
  from epypen import converted, to
  from typing_extensions import Annotated

  @converted
  def _cast(x: int, y: int) -> Annotated[str, to(int)]:
      return str(x + y)

  # The returned string '5' is automatically converted to integer 5.
  assert _cast("2", "3") == 5
  ```

- **`via()`**:  
  Use this helper in parameter annotations to indicate the original type that should be used *before* conversion. It also modifies the function’s signature, so the parameter type in the final signature becomes the type provided to `via()`.
  
  ```python
  from epypen import converted, via, to
  from typing_extensions import Annotated
  import inspect

  @converted
  def foo(x: Annotated[int, via(str)], y: int = 3) -> Annotated[int, to(bool)]:
      return x + y

  # After decoration, the signature of `foo` changes:
  print(inspect.signature(foo))
  # Expected output: (x: str, y: int = 3) -> bool
  ```

> **Note:** Mixing these directives incorrectly (e.g., using `to()` in parameters or `via()` in return types) will raise a `TypeError`.

### Customizing Conversions

**epypen** allows you to customize how conversion is performed by providing your own lists of conversion functions via the `parameter_conversions` and `return_value_conversions` parameters. By default, **epypen** uses predefined conversion functions (e.g., for converting strings to integers, dictionaries to Pydantic models, etc.). However, if you need specialized conversion logic, you can supply your own functions.

For example:

```python
from epypen import converted, ConversionsError

# Custom conversion function that deliberately fails.
def always_fail(typ, obj):
    raise ValueError("Custom conversion failed!")

@converted(parameter_conversions=[always_fail])
def foo(x):
    return x

try:
    foo("test")
except ConversionsError as e:
    print("Custom conversion failed as expected:", e)
```

You can similarly override the conversion logic for return values using `return_value_conversions`. This feature lets you extend **epypen** to handle any custom types or non-standard conversion requirements in your projects.

### Handling Conversion Errors

If no conversion function successfully converts a parameter or a return value, **epypen** raises a `ConversionsError`. This error aggregates all the conversion exceptions, allowing you to see which conversion functions were attempted and why they failed.

```python
from epypen import converted, ConversionsError

def always_fail(typ, obj):
    raise ValueError("Conversion always fails!")

@converted(parameter_conversions=[always_fail])
def foo(x):
    return x

try:
    foo("test")
except ConversionsError as e:
    print("Conversion failed:", e)
```

In this snippet, since every conversion attempt raises an error, a `ConversionsError` is raised containing the list of exceptions from each failed conversion.

---

## Gotchas

- **Annotation Consistency:**  
  Use `via()` for parameters and `to()` for return types. Incorrect usage will result in a `TypeError`.
  
- **Single Directive Limitation:**  
  Only one metadata conversion directive (either `to()` or `via()`) should be provided per parameter or return type to prevent ambiguity.

- **Signature Update:**  
  The function’s signature is modified according to the annotations. If you rely on introspection, verify that the updated signature meets your needs.

---

## Contributing

Contributions are very welcome! If you encounter issues, have suggestions, or want to submit bug fixes, please open an issue or a pull request.

---

## License

This project is licensed under the **GPL-3.0** license.

---

## Acknowledgments

- Inspired by auto-conversion features in FastAPI and Pydantic.

---

Feel free to expand this README with additional examples or troubleshooting tips as **epypen** evolves. Happy converting!