# 🏆 The Project David "Gold Standard" Run Sheet

This document defines the professional engineering workflow for all Project David repositories. It ensures **Security, Quality, and Automation** for a massive user base.

---

## 1. Local Development "The Butler Loop"
Every time you commit code, your local "Butler" (pre-commit) checks your work.

### The Standard Cycle:
1. **Stage your work:** `git add .`
2. **Attempt Commit:** `git commit -m "feat: my new feature"`
3. **Handle Fails:**
   - **If Auto-Fixers fail (Black/Isort):** They have reformatted the code for you.
     - Run `git add .` again to accept the fixes.
     - Run the same `git commit` command again.
   - **If Logic/Security fails (Bandit/Mypy):**
     - Open the file and fix the specific line mentioned.
     - Run `git add .`, then `git commit` again.
4. **The Goal:** A clean, green "Passed" wall before you ever push to GitHub.

---

## 2. Standardized Branching Model
Never work directly on `main`. Use this hierarchy:

| Branch | Purpose | Target | Versioning |
| :--- | :--- | :--- | :--- |
| **`main`** | Production / Stable | Production PyPI | Releases `1.2.0` |
| **`dev`** | Integration / Pre-release | Test PyPI | Releases `1.2.0-dev.1` |
| **`feat/*`** | Feature development | Merge into `dev` | None (CI only) |
| **`fix/*`** | Bug fixes | Merge into `dev` | None (CI only) |

---

## 3. Conventional Commits (SemVer)
Your commit messages control your versioning and changelogs automatically.

*   `feat: ...` → **Minor** bump (e.g., 1.1.0 -> 1.2.0)
*   `fix: ...` → **Patch** bump (e.g., 1.1.0 -> 1.1.1)
*   `feat!: ...` (with `!`) → **Major** bump (e.g., 1.1.0 -> 2.0.0)
*   `chore:`, `docs:`, `style:`, `refactor:` → **No version change**

---

## 4. Surgical Strike: Resolving Conflicts
When `dev` and `main` conflict on the Version or Changelog, use these "CLI-Only" commands to enforce the "Gold Standard":

```powershell
# 1. While on the 'dev' branch, force specific files to match 'main' exactly
git checkout main -- CHANGELOG.md
git checkout main -- pyproject.toml

# 2. Stage and commit the resolution
git add .
git commit -m "chore(ci): resolve conflicts by adopting main gold standard"
```

---

## 5. Security & Quality Gates (CI/CD)
The GitHub Action (`test_tag_release.yml`) is the final "Seal of Approval." It runs:

1. **Lint:** Black (Pinned to `24.1.1`) and Isort.
2. **Types:** Mypy (PEP 561) checking for null-safety and logic.
3. **Security:** Bandit (Static Analysis) and Safety (Vulnerability Scan).
4. **Test:** Pytest across Python 3.10, 3.11, and 3.12.
5. **Release:** Semantic Release generates the tag, the changelog, and pushes to PyPI.

---

## 6. The "Solo Engineer" Pro-Tips
*   **Pinned Versions:** Always use `black==24.1.1` in your local `.pre-commit-config.yaml` AND your GitHub Action to prevent formatting wars.
*   **Null-Safety:** Always use `(variable or "").rstrip()` to prevent `NoneType` crashes.
*   **Type Marker:** Ensure `src/projectdavid/py.typed` exists so IDEs (VS Code/PyCharm) show your type hints to users.
*   **Security Disclosure:** Never open a Public Issue for a bug; use the `SECURITY.md` process to handle it privately via email.

---
**Status:** SDK Housekeeping COMPLETE.
**Next Up:** Project David Platform Overhaul.
