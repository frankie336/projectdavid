import os
import re


def fix_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex looks for: arg_name: type_name = None
    # and replaces it with: arg_name: type_name | None = None
    # It avoids replacing things that are already marked as | None or Optional
    pattern = r"(\w+):\s*([^|\[\s]+)\s*=\s*None"
    replacement = r"\1: \2 | None = None"

    new_content = re.sub(pattern, replacement, content)

    if content != new_content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"✅ Fixed implicit optionals in {path}")


for root, dirs, files in os.walk("src"):
    for file in files:
        if file.endswith(".py"):
            fix_file(os.path.join(root, file))
