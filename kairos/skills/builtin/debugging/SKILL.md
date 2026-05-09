---
name: debugging
version: 1.0.0
description: Systematic debugging approach: reproduce, isolate, fix, verify. Use for any bug or unexpected behavior.
---
# Debugging Skill

## When to use

When the user reports a bug, an error occurs, or behavior is unexpected.

## 4-Phase Approach

### Phase 1: Reproduce
- Get exact steps to reproduce the bug
- Identify the environment (OS, Python version, dependencies)
- Capture the full error message and stack trace

### Phase 2: Isolate
- Narrow down to the minimum code that triggers the bug
- Use binary search: comment out half the code, test, repeat
- Check if it's a regression (did it work before?)

### Phase 3: Fix
- Understand the root cause before writing a fix
- Write a test that fails before the fix, passes after
- Consider if the fix could break other things

### Phase 4: Verify
- Run the full test suite
- Test edge cases related to the fix
- Add the failing case as a regression test

## Common Error Patterns

| Error | Likely Cause |
|-------|-------------|
| ImportError | Missing dependency, circular import |
| AttributeError | Typos, wrong object type |
| KeyError | Missing dict key, wrong key name |
| TypeError | Wrong number/types of arguments |
| TimeoutError | Network issue, infinite loop |

## Tools to use

- `read_file` — Inspect the failing code
- `terminal` — Run the code, check logs
- `search_files` — Find related code
- `patch` — Apply the fix
