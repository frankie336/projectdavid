# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

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
