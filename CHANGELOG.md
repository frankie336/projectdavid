## [1.0.12](https://github.com/frankie336/entitites_sdk/compare/v1.0.11...v1.0.12) (2025-04-06)


### Bug Fixes

* add validators dependency ([2c52f21](https://github.com/frankie336/entitites_sdk/commit/2c52f212035ed9245540d93df064aedf4a2cb7e0))

## [1.0.11](https://github.com/frankie336/entitites_sdk/compare/v1.0.10...v1.0.11) (2025-04-06)


### Bug Fixes

* README.md with correct badge ([a59df73](https://github.com/frankie336/entitites_sdk/commit/a59df73a289e5847d2246686da448ab1d4ad257c))

## [1.0.10](https://github.com/frankie336/entitites_sdk/compare/v1.0.9...v1.0.10) (2025-04-06)


### Bug Fixes

* Add missing dependencies to toml ([5a78cdc](https://github.com/frankie336/entitites_sdk/commit/5a78cdc170390ffcc95f85aba000e9868a7d33db))

## [1.0.9](https://github.com/frankie336/entitites_sdk/compare/v1.0.8...v1.0.9) (2025-04-06)


### Bug Fixes

* _version.py relative import error ([96a5be4](https://github.com/frankie336/entitites_sdk/commit/96a5be4dd5ad85bb158332c7ca86dfe87151af31))

## [1.0.8](https://github.com/frankie336/entitites_sdk/compare/v1.0.7...v1.0.8) (2025-04-06)


### Bug Fixes

* test_tag_release.yml ([53bb318](https://github.com/frankie336/entitites_sdk/commit/53bb3186d60dfc38ba76c3180cc064a3f193d42e))

## [1.0.7](https://github.com/frankie336/entitites_sdk/compare/v1.0.6...v1.0.7) (2025-04-06)


### Bug Fixes

* update workflow to use new trusted publisher and build flow ([1179def](https://github.com/frankie336/entitites_sdk/commit/1179def6e74ef2cbcb4dc570cd76d239ad84e1b2))

## [1.0.6](https://github.com/frankie336/entitites_sdk/compare/v1.0.5...v1.0.6) (2025-04-06)


### Bug Fixes

* align pyproject version to v1.0.5 ([e8d12e0](https://github.com/frankie336/entitites_sdk/commit/e8d12e0e86f46d745a8b8731c7e663180e04c143))

## [1.0.5](https://github.com/frankie336/entitites_sdk/compare/v1.0.4...v1.0.5) (2025-04-06)


### Bug Fixes

* bump version to 1.0.4 ([37650d9](https://github.com/frankie336/entitites_sdk/commit/37650d948585fa3e176016b49dcad2967c83a4f2))
* Test workflow-8 ([cc0c25e](https://github.com/frankie336/entitites_sdk/commit/cc0c25ef60732bd28d5d70ad6554745439124cf4))

## [1.0.4](https://github.com/frankie336/entitites_sdk/compare/v1.0.3...v1.0.4) (2025-04-06)


### Bug Fixes

* Test workflow-3 ([0fb760c](https://github.com/frankie336/entitites_sdk/commit/0fb760c0a3dbc2a7e43256ad891e900808cf0eac))

## [1.0.3](https://github.com/frankie336/entitites_sdk/compare/v1.0.2...v1.0.3) (2025-04-06)


### Bug Fixes

* Test workflow-2 ([cc8730f](https://github.com/frankie336/entitites_sdk/commit/cc8730f290b2b2a3ff10f3fc76092650debcbb5f))

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
