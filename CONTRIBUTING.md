# Contributing to cuda-bench-suite

## Development Setup

```bash
git clone https://github.com/DevSelf12/cuda-bench-suite.git
cd cuda-bench-suite
pip install -e ".[dev]"
```

## Code Style

- Python: [Ruff](https://github.com/astral-sh/ruff) formatter + linter
- CUDA: 4-space indent, K&R braces
- Max line length: 100

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Adding a New Benchmark

See [docs/adding_benchmarks.md](docs/adding_benchmarks.md).

## Pull Request Process

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-benchmark`)
3. Add tests for new functionality
4. Ensure CI passes
5. Open a PR with a clear description

## Reporting Issues

- Include GPU model, CUDA version, driver version
- Include minimal reproducible example
- Attach benchmark output if relevant

## License

By contributing, you agree that your contributions will be licensed under MIT.
