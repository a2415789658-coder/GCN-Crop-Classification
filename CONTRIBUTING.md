# Contributing to GCN Crop Classification

Thank you for your interest in contributing to this project! Every contribution — whether it is a bug fix, new feature, documentation improvement, or suggestion — is valued and appreciated. This guide will help you get started.

---

## Getting Started

1. **Fork the repository** on GitHub.

2. **Clone your fork** locally:

   ```bash
   git clone https://github.com/<your-username>/GCN-Crop-Classification.git
   cd GCN-Crop-Classification
   ```

3. **Create a conda environment** with the required dependencies:

   ```bash
   conda create -n geodl python=3.9 -y
   conda activate geodl
   conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
   conda install pyg -c pyg -y
   conda install rasterio geopandas scikit-learn matplotlib seaborn -c conda-forge -y
   pip install -r requirements.txt
   ```

4. **Create a new branch** for your work:

   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Development Workflow

1. **Make your changes** in your feature branch. Keep commits focused and atomic.

2. **Run flake8 linting** to ensure code quality:

   ```bash
   flake8 . --max-line-length=120 --ignore=E501,W503,E203
   ```

3. **Test that core imports work** correctly:

   ```bash
   python -c "from gcn_crop_classification import GCN"
   ```

4. **Commit with clear, descriptive messages**:

   ```bash
   git add .
   git commit -m "Add: short description of your change"
   ```

5. **Push your branch** to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

---

## Pull Request Process

1. **Update the README** if your changes affect usage, installation, or project structure.
2. **Ensure CI passes** — all linting and import checks must succeed before review.
3. **Use the PR template** if one is provided; otherwise, include a clear summary of your changes.
4. **Request a review** from a maintainer once your PR is ready.

Please be patient during the review process. Maintainers may request changes or ask clarifying questions.

---

## Code Style

- Follow **PEP 8** conventions throughout the codebase.
- **Max line length**: 120 characters.
- Use **descriptive variable names** — prefer `adjacency_matrix` over `am`.
- Add **comments for complex logic**, especially around graph construction and GCN layers.
- Keep functions focused on a single responsibility.
- Use type hints where practical.

---

## Reporting Issues

When reporting a bug or requesting a feature, please use the appropriate GitHub issue template:

- **Bug Report**: Describe the problem, steps to reproduce, expected vs. actual behavior.
- **Feature Request**: Describe the proposed feature and its motivation.

In all cases, include the following environment details:

- Operating system and version
- Python version
- PyTorch and PyG versions
- CUDA version (if applicable)
- Relevant log output or error messages

---

## License

By contributing to this project, you agree that your contributions will be licensed under the [MIT License](LICENSE), the same license that covers this project.
