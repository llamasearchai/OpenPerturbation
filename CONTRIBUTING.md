# Contributing to OpenPerturbation

Thank you for considering contributing to **OpenPerturbation**! We welcome all forms of contribution including bug reports, feature requests, documentation improvements and pull-requests.

## Code of Conduct

This project adopts the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct. By participating you are expected to uphold this code.

## Getting Started

1. **Fork** the repository on GitHub and clone your fork.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the test-suite to verify everything is green:
   ```bash
   pytest -q
   ```

## Branching Model

* **main** – Stable, release-ready code.
* **develop** – Active development.
* **feature/xyz** – New features.
* **fix/xyz** – Bug fixes.

Please branch from **develop** and target your pull-request to **develop**.

## Commit Guidelines

* Use imperative, present-tense: "Add support for X", not "Added".
* Limit the subject line to 72 characters.
* Reference related issues in the body (e.g. "Closes #123").

## Pull-Request Checklist

- [ ] Tests added/updated
- [ ] All tests pass (`pytest -q`)
- [ ] Lint passes (`flake8`)
- [ ] Type checks pass (`pyright`)
- [ ] Documentation updated

## Reporting Bugs

Please open an issue and include:

* Steps to reproduce
* Expected behaviour
* Actual behaviour
* Environment details (OS, Python version, etc.)

## Feature Requests

Open an issue describing the motivation and proposed solution.

## Contact

Questions? Ping **@nikjois** on GitHub or email <nikjois@llamasearch.ai>. 