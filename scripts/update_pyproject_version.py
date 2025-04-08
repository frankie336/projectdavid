# scripts/update_pyproject_versions.py
import re
import sys
from pathlib import Path

def update_version(file_path_str, new_version):
    """Updates the version = "x.y.z" line in a single pyproject.toml file."""
    file_path = Path(file_path_str)
    print(f"Processing: {file_path}...")
    if not file_path.is_file():
        print(f"❌ Error: File not found at {file_path}")
        return False # Indicate failure for this file

    try:
        content = file_path.read_text(encoding="utf-8")
        # Regex to find version = "x.y.z" (allowing variations in spacing and quote types)
        # Uses capture groups to preserve the quote type used
        version_pattern = r'(version\s*=\s*["\'])\d+\.\d+\.\d+(["\'])'
        replacement = rf'\g<1>{new_version}\g<2>' # Use backreferences

        new_content, num_replacements = re.subn(version_pattern, replacement, content, count=1)

        if num_replacements == 0:
            # Handle case where version might be missing or doesn't match expected semver format yet
            print(f"⚠️ Warning: Version pattern not found or matched in {file_path}. Attempting to add.")
            # Attempt to add the version line under [tool.poetry] or [project]
            # This is a basic attempt, might need refinement based on your toml structure
            tool_poetry_pattern = r'(\[tool\.poetry\])'
            project_pattern = r'(\[project\])' # PEP 621 standard

            added = False
            if re.search(tool_poetry_pattern, content):
                 new_content = re.sub(tool_poetry_pattern, rf'\1\nversion = "{new_version}"', content, count=1)
                 added = True
                 print(f"   Attempting to add version under [tool.poetry]")
            elif re.search(project_pattern, content):
                 new_content = re.sub(project_pattern, rf'\1\nversion = "{new_version}"', content, count=1)
                 added = True
                 print(f"   Attempting to add version under [project]")

            if not added:
                 print(f"❌ Error: Could not find [tool.poetry] or [project] section to add version in {file_path}.")
                 return False # Could not add

            # Write the content with the potentially added version
            file_path.write_text(new_content, encoding="utf-8")
            print(f"✅ Added {file_path.name} -> version = \"{new_version}\"")
            return True

        else:
            # Version was found and updated
            file_path.write_text(new_content, encoding="utf-8")
            print(f"✅ Patched {file_path.name} -> version = \"{new_version}\"")
            return True # Indicate success for this file

    except Exception as e:
        print(f"❌ Error updating version file {file_path}: {e}")
        return False # Indicate failure for this file

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("❌ Error: Missing arguments.")
        print("Usage: python scripts/update_pyproject_versions.py <new_version> <path/to/pyproject1.toml> [path/to/pyproject2.toml ...]")
        sys.exit(1)

    new_version_arg = sys.argv[1]
    file_paths_to_update = sys.argv[2:]

    # Basic version format validation
    if not re.match(r"^\d+\.\d+\.\d+$", new_version_arg):
        print(f"❌ Error: Invalid version format received: {new_version_arg}")
        sys.exit(1)

    all_successful = True
    print(f"Attempting to update version to {new_version_arg} in files:")
    for proj_file in file_paths_to_update:
         print(f"- {proj_file}")

    for proj_file in file_paths_to_update:
        if not update_version(proj_file, new_version_arg):
            all_successful = False # Track if any update failed

    if not all_successful:
        print("\n❌ One or more files failed to update.")
        sys.exit(1) # Exit with error code if any update failed

    print("\n✅ All specified pyproject.toml files processed successfully.")
