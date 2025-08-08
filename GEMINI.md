### 🔄 Project Awareness & Context
- **Always read `docs/MODEL_ARC.md` and `ARCHITECTURE_REFACTOR_SUMMARY.md`** at the start of a new conversation to understand the project's architecture, goals, style, and constraints. 
- **Use venv-test** (the virtual environment named 'venv-test') whenever executing Python commands, including for unit tests.

### 🧱 Code Structure & Modularity
- **Never create a file longer than 500 lines of code.** If a file approaches this limit, refactor by splitting it into modules or helper files.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
- **Use clear, consistent imports** (prefer relative imports within packages).
- **Use python_dotenv and load_env()** for environment variables.

### 🧪 Testing & Reliability
- **Always create Pytest unit tests for new features** (functions, classes, routes, etc).
- **After updating any logic**, check whether existing unit tests need to be updated. If so, do it.
- **Tests should live in a `tests` folder** mirroring the main app structure.
  - Include at least:
    - 1 test for expected use
    - 1 edge case
    - 1 failure case

### 📎 Style & Conventions
- **Use Python** as the primary language.
- **Follow PEP8**, use type hints, and format with `black`.
- Write **docstrings for every function** using the Google style:
  ```python
  def example():
      """
      Brief summary.

      Args:
          param1 (type): Description.

      Returns:
          type: Description.
      """
  ```

#### Architecture Violations
- **DON'T** bypass the BaseUIModule pattern for new modules
- **DON'T** duplicate common functionality across modules (use mixins instead)
- **DON'T** create direct dependencies between modules (use SharedMethodRegistry instead)
- **DON'T** implement complex multi-tab interfaces (use single-screen approach)

#### Performance and Memory
- **DON'T** create memory leaks by not cleaning up event handlers
- **DON'T** hold references to large objects in module-level variables
- **DON'T** initialize heavy operations during module import
- **DON'T** block the UI thread with long-running operations


### 📚 Documentation & Explainability
- **Comment non-obvious code** and ensure everything is understandable to a mid-level developer.
- When writing complex logic, **add an inline `# Reason:` comment** explaining the why, not just the what.

### 🧠 AI Behavior Rules
- **Never assume missing context. Ask questions if uncertain.**
- **Never hallucinate libraries or functions** – only use known, verified Python packages.
- **Always confirm file paths and module names** exist before referencing them in code or tests.
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a task from `TASK.md`.