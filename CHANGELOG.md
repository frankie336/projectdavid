## [1.0.2](https://github.com/frankie336/entitites_sdk/compare/v1.0.1...v1.0.2) (2025-04-06)


### Bug Fixes

* Test workflow ([afc8e6b](https://github.com/frankie336/entitites_sdk/commit/afc8e6b4e036baa5f4a66a5bf8bed62c2ec2fde7))

## [1.0.1](https://github.com/frankie336/entitites_sdk/compare/v1.0.0...v1.0.1) (2025-04-06)


### Bug Fixes

* entities_common version issue again ([6dc6c45](https://github.com/frankie336/entitites_sdk/commit/6dc6c4500c81e61278bdb0254881cc1dfc537798))

# 1.0.0 (2025-04-06)


### Bug Fixes

* Fix auto release ([a9a1b2e](https://github.com/frankie336/entitites_sdk/commit/a9a1b2e0d03a707be0510e171fd57cb0c3c7d5f2))
* Require latest entities_common in toml ([6ca402b](https://github.com/frankie336/entitites_sdk/commit/6ca402b0532946eef68e93862324d281e181cc39))
* resolve entities_common version issue ([6b64ef6](https://github.com/frankie336/entitites_sdk/commit/6b64ef6bdde7f21245a728d106d3f95daa1422b9))


### Features

* add support for auto version tagging ([5ea9aed](https://github.com/frankie336/entitites_sdk/commit/5ea9aed79fa4f37789c463458409126d60da2388))

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.1] - 2025-04-05

### Added
- Trusted publishing setup for PyPI and TestPyPI, including GitHub Actions workflow with tag-based trigger.
- `scripts/pin_entities_common.py`: utility to pin latest commit SHA from `entities_common` into `pyproject.toml` and `requirements.txt`.
- CI workflow `pin-dependencies.yml` that auto-pins `entities_common` on each push to `main`.

### Fixed
- Flake8 linting issues across `file_processor.py` due to missing typing imports.
- `LiteralString` fallback import for Python < 3.11 environments.
- Typos and inconsistencies in GitHub workflow tags (`test-v*` vs `v*`) that prevented job execution.

### Changed
- Replaced dynamic `entities_common` Git dependency with pinned SHA references.
- Made the `publish` workflow fully conformant with [Trusted Publishing](https://docs.pypi.org/trusted-publishers/).



## [0.3.0] - 2025-04-04

### Added
- Introduced `RunMonitorClient` with full lifecycle event handling for assistant runs.
- Added `EntitiesInternalInterface` as a unified internal service orchestrator.
- `ActionsClient`, `MessagesClient`, `RunsClient`, and `VectorStoreClient` now wrapped and lazy-loaded under `Entities(...)`.
- Support for tool invocation streaming with `on_action_required`, `on_tool_invoked`, and `on_complete` callbacks.
- `code_interpreter_stream` and `file_download_url` support in SSE stream parsing.

### Changed
- Moved `EntitiesEventHandler` logic from Flask backend into internal API and SDK boundary.
- SDK now defaults to `SDK_VERSION = "0.3.0"`.

---

## [0.2.0] - 2025-03-01

### Added

---

## [0.1.0-alpha] - 2025-01-15

### Added
- Core SDK skeleton: `Entities`, `UsersClient`, `MessagesClient`, etc.
- Basic message submission and tool output lifecycle.
- Initial assistant threading and function call support.
