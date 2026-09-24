# How to contribute

We'd love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Setting up your development environment

### Prerequisites

*   Python 3.10 or later (Python 3.13 recommended)
*   [JAX](https://docs.jax.dev/en/latest/installation.html) 0.7.2 or later
*   For GPU development: CUDA 13+ and a supported NVIDIA GPU (SM80+)

### Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/openxla/tokamax.git
cd tokamax
pip install -e ".[dev]"
```

### Running tests

Tests can be run using pytest:

```bash
# Run all tests
pytest tokamax/

# Run tests for a specific module
pytest tokamax/_src/ops/attention/api_test.py

# Run a specific test
pytest tokamax/_src/ops/attention/api_test.py::AttentionApiTest::test_basic
```

> **Note:** Some tests require GPU or TPU hardware and will be skipped on CPU.
> Tests use `jax_enable_x64=False` by default (configured in `conftest.py`).

## Contribution process

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

### Coding style

*   Follow the existing code style in the repository.
*   Use type annotations — Tokamax uses
    [jaxtyping](https://github.com/patrick-kidger/jaxtyping) for array shape
    annotations.
*   Add docstrings to public functions following
    [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

### Testing

*   Add tests for any new functionality.
*   Ensure existing tests pass before submitting a PR.
*   Use the test base classes (e.g., `test_base.py`) when adding tests for
    new operation implementations.

### Pull request guidelines

*   Keep PRs focused — one logical change per PR.
*   Write clear commit messages explaining *why* the change was made.
*   Reference any related issues in the PR description.
*   Respond promptly to code review feedback.
