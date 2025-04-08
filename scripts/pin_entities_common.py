import subprocess
from pathlib import Path
import re

REPO = "https://github.com/frankie336/entities_common.git"


def get_latest_commit_sha() -> str:
    result = subprocess.run(
        ["git", "ls-remote", REPO, "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git command failed: {result.stderr.strip()}")

    output = result.stdout.strip()
    if not output:
        raise ValueError("No output received from git ls-remote.")
    return output.split()[0]


def update_pyproject_toml(sha: str):
    path = Path("pyproject.toml")
    if not path.exists():
        print("pyproject.toml not found.")
        return

    contents = path.read_text()
    pinned = f"entities_common @ git+{REPO}@{sha}"

    updated = re.sub(
        r'entities_common\s*@\s*git\+https://github\.com/frankie336/entities_common\.git(@[^\s"]+)?',
        pinned,
        contents,
    )

    path.write_text(updated)
    print(f"✅ Updated `pyproject.toml` with pinned SHA: {sha}")


def update_requirements_txt(sha: str):
    path = Path("requirements.txt")
    if not path.exists():
        return

    pinned_modern = f"entities_common @ git+{REPO}@{sha}"
    pinned_legacy = f"git+{REPO}@{sha}#egg=entities_common"

    lines = path.read_text().splitlines()
    updated_lines = []
    for line in lines:
        if "entities_common @ git+" in line:
            updated_lines.append(pinned_modern)
        elif (
            "git+https://github.com/frankie336/entities_common.git" in line
            and "#egg=entities_common" in line
        ):
            updated_lines.append(pinned_legacy)
        else:
            updated_lines.append(line)

    path.write_text("\n".join(updated_lines))
    print(f"✅ Updated `requirements.txt` with pinned SHA: {sha}")


if __name__ == "__main__":
    sha = get_latest_commit_sha()
    update_pyproject_toml(sha)
    update_requirements_txt(sha)
