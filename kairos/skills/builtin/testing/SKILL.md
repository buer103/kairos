---
name: testing
version: 1.0.0
description: Write and run tests: unit tests, integration tests, edge cases, and test-driven development workflows.
---
# Testing Skill

## When to use

When the user asks to write tests, run tests, fix failing tests, or follow TDD.

## Test-Driven Development (RED-GREEN-REFACTOR)

1. **RED** — Write a failing test that defines the expected behavior
2. **GREEN** — Write the minimum code to make the test pass
3. **REFACTOR** — Clean up while keeping tests green

## Test Structure

### Python (pytest)
```python
import pytest
from mymodule import function_to_test

def test_happy_path():
    result = function_to_test("valid input")
    assert result == expected

def test_edge_cases():
    assert function_to_test("") == default_value
    assert function_to_test(None) is None

def test_error_handling():
    with pytest.raises(ValueError):
        function_to_test(invalid_input)
```

## What to test

1. **Happy path** — The main use case works
2. **Edge cases** — Empty inputs, max values, boundary conditions
3. **Error states** — Invalid inputs, missing dependencies, timeouts
4. **Integration** — Components work together correctly
5. **Regression** — Previously fixed bugs stay fixed

## Pitfalls

- Don't test implementation details — test behavior
- Avoid test interdependence — each test should run independently
- Mock external services, not your own code
- Run tests with coverage: `pytest --cov=kairos tests/`
