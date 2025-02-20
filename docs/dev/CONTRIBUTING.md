# 🤝 Contribution Guide

## 📋 Overview

Panduan ini menjelaskan cara berkontribusi ke project SmartCash.

## 🚀 Getting Started

### 1. Setup Development Environment
```bash
# Clone repository
git clone https://github.com/yourusername/smartcash.git
cd smartcash

# Create virtual environment
conda create -n smartcash python=3.9
conda activate smartcash

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Pre-commit Setup
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run all hooks
pre-commit run --all-files
```

## 🔄 Development Workflow

### 1. Create Issue
- Check existing issues
- Create new issue with clear description
- Wait for issue to be assigned

### 2. Create Branch
```bash
# Create feature branch
git checkout -b feature/issue-number-description

# Create bugfix branch
git checkout -b bugfix/issue-number-description
```

### 3. Make Changes
- Follow code style guide
- Add tests
- Update documentation

### 4. Commit Changes
```bash
# Stage changes
git add .

# Commit with conventional commits
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug"
git commit -m "docs: update documentation"
```

### 5. Submit PR
- Push changes
- Create pull request
- Link issue
- Wait for review

## 📝 Code Style

### 1. Python Style
- Follow PEP 8
- Use type hints
- Max line length: 88
- Use docstrings

### 2. Documentation Style
- Clear and concise
- Include examples
- Update README
- Keep docs up-to-date

## ✅ Testing

### 1. Write Tests
```python
def test_detection():
    """Test object detection."""
    model = SmartCash()
    image = load_test_image()
    result = model.detect(image)
    
    assert len(result) > 0
    assert result[0]["confidence"] > 0.5
```

### 2. Run Tests
```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_detection.py

# Run with coverage
pytest --cov=smartcash
```

## 📊 Code Review

### 1. Review Checklist
- Code style
- Test coverage
- Documentation
- Performance
- Security

### 2. Review Process
1. Read description
2. Check changes
3. Run tests
4. Review code
5. Provide feedback

## 🔒 Security

### 1. Guidelines
- No secrets in code
- Use environment variables
- Validate inputs
- Handle errors

### 2. Reporting Issues
- Use security template
- Private disclosure
- Wait for response

## 📈 Performance

### 1. Profiling
```bash
# Profile code
python -m cProfile -o output.prof script.py

# Analyze results
snakeviz output.prof
```

### 2. Benchmarking
```bash
# Benchmark function
python -m timeit -s "from script import func" "func()"
```

## 📚 Documentation

### 1. Code Documentation
```python
def process_image(
    image: np.ndarray,
    size: Tuple[int, int] = (640, 640)
) -> np.ndarray:
    """
    Process image for model input.
    
    Args:
        image: Input image
        size: Target size
        
    Returns:
        Processed image
    """
    return cv2.resize(image, size)
```

### 2. Project Documentation
- README.md
- API documentation
- User guides
- Technical docs

## 🐛 Bug Reports

### 1. Template
```markdown
## Description
Clear description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: Ubuntu 20.04
- Python: 3.9.7
- Package version: 1.0.0
```

### 2. Debug Information
- Error messages
- Stack traces
- Screenshots
- System info

## 🎯 Feature Requests

### 1. Template
```markdown
## Problem
What problem does this solve?

## Proposed Solution
How should it be solved?

## Alternatives
What alternatives exist?

## Additional Context
Any other information
```

### 2. Acceptance Criteria
- Clear requirements
- Test cases
- Performance metrics
- Documentation needs

## 🔄 Version Control

### 1. Branch Strategy
- main: production
- develop: development
- feature/*: features
- bugfix/*: bug fixes

### 2. Version Tags
```bash
# Create version tag
git tag -a v1.0.0 -m "Release v1.0.0"

# Push tag
git push origin v1.0.0
```

## 📦 Release Process

### 1. Preparation
- Update version
- Update changelog
- Run tests
- Build docs

### 2. Release Steps
1. Merge to main
2. Create tag
3. Build package
4. Publish release

## 🚀 Next Steps

1. [Git Workflow](GIT_WORKFLOW.md)
2. [Testing Guide](TESTING.md)
3. [Code Style](CODE_STYLE.md)
