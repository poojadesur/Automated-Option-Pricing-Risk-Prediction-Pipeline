# Contributing to Option Pricing Pipeline

Thank you for your interest in contributing to the Automated Option Pricing & Risk Prediction Pipeline!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/Automated-Option-Pricing-Risk-Prediction-Pipeline.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Install in development mode: `pip install -e .`
7. Install development dependencies: `pip install pytest pytest-cov black flake8 isort`

## Development Workflow

1. Create a new branch for your feature: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Format code: `black src/`
5. Sort imports: `isort src/`
6. Check linting: `flake8 src/`
7. Commit your changes: `git commit -m "Description of changes"`
8. Push to your fork: `git push origin feature/your-feature-name`
9. Create a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Add type hints where appropriate

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for >90% code coverage
- Include both unit tests and integration tests

## Documentation

- Update README.md for significant changes
- Add docstrings to new functions/classes
- Update configuration examples if needed
- Add usage examples for new features

## Pull Request Process

1. Ensure your PR description clearly describes the problem and solution
2. Link any related issues
3. Ensure all tests pass
4. Request review from maintainers
5. Address any review comments

## Code Review Criteria

- Code quality and style
- Test coverage
- Documentation completeness
- Performance considerations
- Security implications

## Questions?

Open an issue for any questions or concerns.
