import re
import sys
from pathlib import Path

def update_version(file_path, new_version):
    content = Path(file_path).read_text(encoding="utf-8")
    new_content = re.sub(
        r'version\s*=\s*"[0-9]+\.[0-9]+\.[0-9]+"',
        f'version = \"{new_version}\"',
        content,
    )
    Path(file_path).write_text(new_content, encoding="utf-8")
    print(f"🔧 Patched pyproject.toml → version = {new_version}")

if __name__ == "__main__":
    _, version = sys.argv
    update_version("pyproject.toml", version)
