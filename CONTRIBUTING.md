# Contributing to KnoAD-Net

First off, thank you for considering contributing to KnoAD-Net! 🎉

This document provides guidelines for contributing to this Alzheimer's Disease detection research project. Following these guidelines helps maintain code quality, ensures reproducibility, and fosters a welcoming community.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)

---

## 📜 Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of:
- Experience level
- Background or identity
- Geographic location
- Age, gender, or any other personal characteristic

---

## 🤝 How Can I Contribute?

### Reporting Bugs

Before creating a bug report:
1. **Check existing issues** to avoid duplicates
2. **Reproduce the bug** to ensure it's consistent
3. **Gather information**: OS, Python version, PyTorch version, error messages

**When submitting a bug report, include:**
- Clear, descriptive title
- Detailed steps to reproduce
- Expected vs. actual behavior
- Full error messages and stack traces
- System information (OS, Python version, GPU/CPU)
- Screenshots (if applicable)

**Example:**
```markdown
**Bug Description:** Model fails to load checkpoint on CPU-only systems

**Steps to Reproduce:**
1. Install requirements on system without GPU
2. Run `python notebook3_knoadnet_core.py`
3. Error occurs at line 123

**Expected:** Model loads successfully on CPU
**Actual:** RuntimeError: CUDA not available

**Environment:**
- OS: Ubuntu 22.04
- Python: 3.9.7
- PyTorch: 2.0.1
```

### Suggesting Enhancements

We welcome suggestions for:
- New features or capabilities
- Performance improvements
- Better documentation
- Enhanced visualizations
- Additional evaluation metrics

**When suggesting enhancements:**
1. **Check existing issues** for similar suggestions
2. **Provide clear use case** - why is this valuable?
3. **Include implementation ideas** (if you have them)
4. **Consider scope** - does it align with project goals?

### Contributing Code

We appreciate code contributions! Areas where contributions are especially welcome:

#### High Priority
- 🐛 Bug fixes for known issues
- 📝 Documentation improvements
- ✅ Additional test coverage
- 🔧 Performance optimizations

#### Medium Priority
- 🎨 Visualization enhancements
- 📊 New evaluation metrics
- 🧪 Data augmentation techniques
- 🔍 Attention mechanism improvements

#### Advanced Contributions
- 🧠 Alternative model architectures
- 🌐 Multi-dataset support
- 🔄 Longitudinal analysis features
- 🚀 Deployment tools and APIs

---

## 🔧 Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv or conda)

### Setup Steps

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/Alzimers-Research.git
cd Alzimers-Research

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install development dependencies (if we add them)
pip install black flake8 pytest pytest-cov

# 5. Create a branch for your work
git checkout -b feature/your-feature-name
```

### Verify Installation

```bash
# Test imports
python -c "import torch, nibabel, transformers; print('✓ Installation successful!')"

# Run a quick test (if applicable)
python test.py
```

---

## 🔄 Contribution Workflow

### 1. Create an Issue First

For significant changes:
1. Create an issue describing your proposed change
2. Wait for feedback from maintainers
3. Proceed with implementation once approved

### 2. Branch Naming Convention

Use descriptive branch names:
- `feature/add-3d-cnn-support`
- `bugfix/fix-memory-leak-in-dataloader`
- `docs/update-installation-guide`
- `refactor/improve-rag-module`

### 3. Make Your Changes

- Write clear, documented code
- Follow existing code style
- Add tests for new features
- Update documentation as needed
- Keep changes focused and atomic

### 4. Test Your Changes

```bash
# Run existing tests
python -m pytest tests/

# Test specific functionality manually
python phase1_data_pipeline.py  # If you modified data pipeline
python notebook3_knoadnet_core.py  # If you modified model
```

### 5. Commit Your Changes

Follow [Commit Message Guidelines](#commit-message-guidelines)

```bash
git add .
git commit -m "feat: add 3D CNN support for volumetric MRI"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

---

## 💻 Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Organized by standard library, third-party, local

### Code Formatting

We use **Black** for code formatting:

```bash
# Format all Python files
black .

# Check formatting without modifying
black --check .
```

### Linting

We use **Flake8** for linting:

```bash
# Run linting
flake8 . --max-line-length=88 --extend-ignore=E203,W503
```

### Type Hints

Use type hints for function signatures:

```python
def preprocess_mri(image: np.ndarray, target_size: int = 128) -> torch.Tensor:
    """
    Preprocess MRI image.
    
    Args:
        image: Input MRI image array
        target_size: Target dimension for resizing
        
    Returns:
        Preprocessed image tensor
    """
    # Implementation
    pass
```

### Documentation Strings

Use Google-style docstrings:

```python
def train_model(model, train_loader, val_loader, num_epochs=40):
    """
    Train the KnoAD-Net model.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int, optional): Number of training epochs. Defaults to 40.
        
    Returns:
        dict: Training history with loss and accuracy metrics
        
    Raises:
        ValueError: If model is None or dataloaders are empty
    """
    pass
```

### Code Organization

- **Keep functions small**: One function should do one thing
- **Use meaningful names**: `calculate_attention_scores()` not `calc()`
- **Add comments**: Explain *why*, not *what*
- **Avoid magic numbers**: Use constants

```python
# Good
IMAGE_SIZE = 128
DROPOUT_RATE = 0.5

# Not so good
img = resize(img, 128)  # What is 128?
```

---

## ✅ Testing Guidelines

### Writing Tests

- Test new features thoroughly
- Include edge cases
- Test error handling
- Use meaningful test names

```python
def test_dataloader_handles_empty_dataset():
    """Test that dataloader raises appropriate error for empty dataset."""
    # Test implementation
    pass

def test_model_output_shape_correct():
    """Test that model output has expected shape (batch_size, 3)."""
    # Test implementation
    pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::test_model_output_shape_correct
```

---

## 📝 Documentation Standards

### Code Documentation

- Every module should have a docstring
- Every public function/class should have a docstring
- Complex algorithms should have inline comments
- Update docs when changing functionality

### Markdown Documentation

- Use clear headings and structure
- Include code examples
- Add screenshots for visual changes
- Keep language clear and concise

### Updating Documentation

When you make changes, update:
- **README.md**: If you add features or change installation
- **docs/PROJECT_DOCUMENTATION.md**: For technical details
- **CHANGELOG.md**: Add entry for your changes
- **Inline comments**: Update if logic changes

---

## 📋 Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting, missing semicolons, etc. (no code change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```bash
# Feature addition
feat(model): add 3D CNN support for volumetric MRI

Implements 3D convolutional layers to process entire brain volumes
instead of 2D slices. Includes:
- 3D convolution backbone
- Modified data loader for 3D input
- Updated training loop

Closes #123

# Bug fix
fix(dataloader): resolve memory leak in batch processing

Fixed issue where MRI tensors were not released after batch processing,
causing memory to accumulate during training.

Fixes #456

# Documentation
docs(readme): update installation instructions for M1 Macs

Added specific instructions for Apple Silicon users including
Rosetta 2 requirements and ARM-compatible PyTorch installation.
```

### Best Practices

- **Use imperative mood**: "add feature" not "added feature"
- **Keep subject line under 50 characters**
- **Capitalize subject line**
- **No period at end of subject**
- **Include detailed body** for complex changes
- **Reference issues**: "Closes #123" or "Fixes #456"

---

## 🔀 Pull Request Process

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with main
- [ ] No merge conflicts

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
Describe how you tested your changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings

## Related Issues
Closes #123
Related to #456

## Screenshots (if applicable)
Add screenshots for UI/visualization changes
```

### Review Process

1. **Automated checks**: CI/CD runs tests and linting
2. **Maintainer review**: Code review by project maintainers
3. **Feedback**: Address any requested changes
4. **Approval**: PR approved by maintainer(s)
5. **Merge**: PR merged into main branch

### After Merge

- Delete your feature branch
- Update your local main branch
- Close related issues

---

## 🎯 Areas Needing Contribution

### Current Priorities

1. **Testing**: Expand test coverage for all modules
2. **Documentation**: Add more examples and tutorials
3. **Performance**: Optimize data loading and preprocessing
4. **Visualization**: Enhance attention maps and result visualizations

### Longer-Term Goals

1. **3D Support**: Full volumetric MRI processing
2. **Multi-Dataset**: Support for ADNI, AIBL datasets
3. **Deployment**: Web interface and API development
4. **Interpretability**: Enhanced explainability features

---

## ❓ Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Create an issue
- **Feature requests**: Create an issue with "enhancement" label
- **Security issues**: Report privately (see SECURITY.md if available)

---

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Acknowledged in academic publications (for significant contributions)

---

## 📚 Additional Resources

- [PyTorch Contributing Guide](https://pytorch.org/docs/stable/community/contribution_guide.html)
- [Vision Transformers Paper](https://arxiv.org/abs/2010.11929)
- [OASIS Dataset Documentation](https://www.oasis-brains.org/)
- [Medical AI Best Practices](https://www.nature.com/articles/s41591-020-01197-2)

---

**Thank you for contributing to KnoAD-Net! Together, we're advancing AI for Alzheimer's detection.** 🧠✨
