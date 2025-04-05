# Versioning and Release Strategy

This repository uses **semantic-release** for automated versioning and changelog generation, powered by **Conventional Commits**.

---

## 🚀 How It Works

Each push to `main` (or `master`) is scanned for semantic commit messages. Based on the **commit prefix**, the CI system:

1. Determines the next version.
2. Updates `CHANGELOG.md` with generated release notes.
3. Updates `pyproject.toml` with the new version.
4. Tags the commit (`vX.Y.Z`).
5. Optionally publishes to PyPI/TestPyPI.

---

##  Commit Guidelines (Conventional Commits)

Use the following prefixes in **your commit messages** to trigger version bumps:

| Prefix               | Description                          | Version Bump |
|---------------------|--------------------------------------|--------------|
| \`fix:\`             | A bug fix                            | **Patch**    |
| \`feat:\`            | A new feature                        | **Minor**    |
| \`BREAKING CHANGE:\` | Incompatible API change              | **Major**    |
| \`chore:\`           | Build tasks, dependencies, etc.      | None         |
| \`docs:\`            | Documentation only                   | None         |
| \`refactor:\`        | Code changes not fixing or adding    | None         |
| \`test:\`            | Tests only                           | None         |
| \`style:\`           | Formatting, white-space, etc.        | None         |

> ✅ Semantic-release will ignore commits **without a recognized prefix**.

---

## 💻 Local Commit Workflow

We recommend using **[Commitizen](https://github.com/commitizen/cz-cli)** for standardized commit prompts.

### 1. Install

```bash
pip install commitizen
```

### 2. Example Commit Messages

Below are examples of properly formatted commits:

```bash
feat: add support for batch document processing

fix: resolve crash when uploading empty file

docs: update README with installation guide

refactor: simplify embedding chunking logic

BREAKING CHANGE: restructure vector store API contract
```

> Use \`cz commit\` after installing Commitizen for interactive commit prompts that follow this format.
