# Contributing to Web MCP Server

First off, thanks for taking the time to contribute! 🎉

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**Great Bug Reports** include:
- A clear, descriptive title
- Steps to reproduce the behavior
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or screenshots

### 💡 Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues.

**Great Enhancement Suggestions** include:
- A clear, descriptive title
- Step-by-step description of the suggested enhancement
- Explanation of why this would be useful
- Examples of how it would work

### 🔧 Pull Requests

1. **Fork** the repo and create your branch from `main`
2. **Install** dependencies: `pip install -r requirements.txt`
3. **Make** your changes
4. **Test** your changes: `pytest`
5. **Lint** your code: `ruff check . && ruff format .`
6. **Commit** with a clear message
7. **Push** and open a PR

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/web-mcp.git
cd web-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set up pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

## Style Guide

### Python

- Follow [PEP 8](https://pep8.org/)
- Use [ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Write docstrings for public functions (Google style)
- Add type hints

### Commits

- Use clear, descriptive commit messages
- Start with a verb: "Add", "Fix", "Update", "Remove"
- Reference issues when applicable: "Fix #123"

### Tests

- Write tests for new features
- Ensure existing tests pass
- Aim for meaningful coverage, not 100%

## Project Structure

```
web-mcp/
├── app.py              # Main Gradio application
├── rag.py              # RAG pipeline
├── reranker.py         # Cross-encoder reranker
├── analytics.py        # Usage tracking
├── config.py           # Configuration
├── benchmark.py        # Performance benchmarks
├── tests/              # Test suite
│   ├── conftest.py     # Shared fixtures
│   ├── test_*.py       # Test modules
├── .github/
│   └── workflows/      # CI/CD pipelines
├── requirements.txt    # Dependencies
├── Dockerfile          # Container build
└── README.md           # Documentation
```

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test
pytest tests/test_rag.py -v

# Skip slow tests
pytest -m "not slow"
```

## Questions?

Feel free to open an issue for any questions!

---

Thank you for contributing! 🙏
