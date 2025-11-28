# Contributing to c50py

Thank you for your interest in contributing to `c50py`! We welcome contributions from the community to help improve this library.

## How to Contribute

### Reporting Bugs

If you encounter any bugs or issues, please open an issue on GitHub. Include as much detail as possible:
- A clear description of the issue.
- Steps to reproduce the bug (including code snippets).
- Expected behavior vs. actual behavior.
- Your environment details (OS, Python version, `c50py` version).

### Suggesting Enhancements

We appreciate suggestions for new features or improvements. Please open an issue on GitHub describing your idea and how it would benefit the project.

### Pull Requests

1.  **Fork the repository** and create a new branch for your feature or fix.
2.  **Write code** following the existing style and conventions.
3.  **Add tests** to cover your changes.
4.  **Run tests** locally to ensure everything works as expected.
5.  **Submit a Pull Request** (PR) with a clear description of your changes.

## Development Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/daviddiazsolis/c50py.git
    cd c50py
    ```

2.  Install dependencies and the package in editable mode:
    ```bash
    pip install -e .[dev]
    ```
    (Note: `[dev]` assumes you have defined optional dependencies for development in `pyproject.toml`. If not, just `pip install -e .` and install `pytest` separately.)

3.  Run tests:
    ```bash
    pytest
    ```

## Code Style

- We follow PEP 8 guidelines.
- Please ensure your code is clean and well-documented.

## License

By contributing to `c50py`, you agree that your contributions will be licensed under the MIT License.
