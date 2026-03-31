# Contributing to Scan2Sheet

Thank you for your interest in contributing! This document outlines the process for reporting issues, requesting features, and submitting code changes.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Commit Message Convention](#commit-message-convention)
- [Code Style](#code-style)

---

## Code of Conduct

Please be respectful and constructive in all interactions. This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct.

---

## Reporting Bugs

Before filing a bug report, please check existing issues to avoid duplicates.

When filing a bug, include:
- **Python version** and **OS**
- **Steps to reproduce** the issue
- **Expected vs actual behaviour**
- **Sample image** if relevant (anonymize if needed)
- **Full error traceback**

---

## Feature Requests

Open an issue with the label `enhancement` and describe:
- The use case that motivates the feature
- How you envision it working
- Any relevant examples or references

---

## Development Setup

```bash
git clone https://github.com/kratos999-athena/Scan2Sheet.git
cd Scan2Sheet
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

Run tests before making changes to verify baseline:
```bash
pytest tests/ -v
```

---

## Pull Request Process

1. Fork the repo and create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make focused, well-scoped changes.
3. Add or update tests for any changed behaviour in `tests/`.
4. Ensure all tests pass: `pytest tests/ -v`
5. Update `CHANGELOG.md` under the `[Unreleased]` section.
6. Open a Pull Request with a clear description of the change and why it's needed.

---

## Commit Message Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add PDF multi-page input support
fix: correct cell padding in crop_each_bounding_box_and_ocr
docs: update README installation steps
refactor: split preprocessing.py into orientation and skew modules
test: add unit tests for LineExtractor
chore: update requirements.txt versions
```

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for all Python code.
- Use descriptive variable and function names.
- Add docstrings to all public functions and classes (NumPy docstring style preferred).
- Keep functions focused — one responsibility per function.
- Avoid hardcoded paths; use function parameters or config instead.