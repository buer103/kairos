---
name: code-review
version: 1.0.0
description: Review code for quality, security, style, and correctness. Use before merging PRs or committing significant changes.
---
# Code Review Skill

## When to use

When the user asks to review code, review a PR, check code quality, or before
committing significant changes.

## Review Checklist

1. **Correctness** — Does the code do what it claims? Check edge cases.
2. **Security** — SQL injection, XSS, path traversal, hardcoded secrets.
3. **Style** — Consistent with project conventions. PEP 8 for Python.
4. **Performance** — Obvious N+1 queries, unnecessary allocations, blocking I/O.
5. **Testability** — Is the code structured to be testable?
6. **Error handling** — Are exceptions caught appropriately? Error messages helpful?

## Procedure

1. Read the changed files with `read_file`
2. Run existing tests with `terminal`
3. Check for common pitfalls per language
4. Report findings organized by severity: 🔴 Critical / 🟡 Warning / 🔵 Suggestion
5. Offer concrete fix suggestions, not just criticism

## Pitfalls to watch

- Python: mutable default args, bare except, f-string in logging
- JavaScript: `==` vs `===`, callback hell, unhandled promise rejections
- General: hardcoded credentials, missing input validation, race conditions
