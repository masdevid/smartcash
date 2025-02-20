# 🔄 Git Workflow

## 📋 Overview

Dokumen ini menjelaskan workflow Git yang digunakan dalam pengembangan SmartCash.

## 🌳 Branch Structure

### 1. Main Branches
```
main ────────────────────────────────
  │
  ├── develop ────────────────────────
  │     │
  │     ├── feature/add-new-model
  │     │
  │     ├── bugfix/fix-detection
  │     │
  │     └── hotfix/security-patch
  │
  └── release/v1.0.0
```

### 2. Branch Types
- `main`: Production code
- `develop`: Development code
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical fixes
- `release/*`: Release preparation

## 🔄 Workflow Steps

### 1. Feature Development
```bash
# Create feature branch
git checkout develop
git pull origin develop
git checkout -b feature/add-new-model

# Make changes
git add .
git commit -m "feat: add new model architecture"

# Update from develop
git fetch origin develop
git rebase origin/develop

# Push changes
git push origin feature/add-new-model
```

### 2. Bug Fixes
```bash
# Create bugfix branch
git checkout develop
git pull origin develop
git checkout -b bugfix/fix-detection

# Fix bug
git add .
git commit -m "fix: resolve detection issue"

# Push changes
git push origin bugfix/fix-detection
```

### 3. Release Process
```bash
# Create release branch
git checkout develop
git checkout -b release/v1.0.0

# Version bump
bump2version minor
git add .
git commit -m "chore: bump version to 1.0.0"

# Merge to main
git checkout main
git merge release/v1.0.0 --no-ff
git tag -a v1.0.0 -m "Release v1.0.0"

# Merge back to develop
git checkout develop
git merge release/v1.0.0 --no-ff
```

## 📝 Commit Messages

### 1. Conventional Commits
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### 2. Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

### 3. Examples
```bash
feat(model): add EfficientNet backbone
fix(detection): resolve confidence threshold issue
docs(api): update API documentation
style(code): format according to PEP 8
refactor(dataset): restructure data loading
test(evaluation): add mAP tests
chore(deps): update dependencies
```

## 🔍 Code Review

### 1. Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual tests

## Checklist
- [ ] Tests added
- [ ] Documentation updated
- [ ] Code follows style guide
- [ ] All tests passing
```

### 2. Review Process
1. Code review
2. Test review
3. Documentation review
4. Performance review

## 🔄 Version Control

### 1. Git LFS
```bash
# Setup LFS
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.onnx"

# Verify tracking
git lfs ls-files
```

### 2. Git Hooks
```bash
# Pre-commit
#!/bin/sh
pytest
black .
flake8 .

# Pre-push
#!/bin/sh
pytest --cov=smartcash
```

## 📊 Branch Protection

### 1. Main Branch
- Require pull request
- Require review
- Require tests
- No direct push

### 2. Develop Branch
- Require pull request
- Require one review
- Require tests
- Allow rebase

## 🔒 Security

### 1. Secrets
- Use environment variables
- No hardcoded credentials
- Regular key rotation
- Secure storage

### 2. Access Control
- Team permissions
- Branch restrictions
- Deploy keys
- Review requirements

## 📈 CI/CD

### 1. GitHub Actions
```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
    - name: Run tests
      run: |
        pip install -r requirements.txt
        pytest
```

### 2. Release Workflow
```yaml
name: Release

on:
  push:
    tags:
      - "v*"

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Build
      run: |
        python setup.py sdist bdist_wheel
    - name: Release
      uses: softprops/action-gh-release@v1
```

## 📚 Documentation

### 1. Version Documentation
- CHANGELOG.md
- Release notes
- Migration guides
- API changes

### 2. Branch Documentation
- Branch purpose
- Merge strategy
- Protection rules
- Naming convention

## 🚀 Best Practices

1. Regular commits
2. Clear messages
3. Branch management
4. Code review
5. Version control
6. Documentation
7. Testing
8. Security

## 📈 Next Steps

1. [Testing Guide](TESTING.md)
2. [Code Style](CODE_STYLE.md)
3. [Release Process](RELEASE.md)
