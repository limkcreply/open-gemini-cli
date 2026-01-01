# Implementation Plan - Refactor `read_many_files`

## Phase 1: Analysis and Reproduction
- [ ] Task: Create a performance benchmark script that generates a large number of temporary files (e.g., 1000+) and measures the execution time and memory usage of the current `read_many_files` implementation.
- [ ] Task: Analyze the current implementation in `packages/core/src/tools/read_many_files` (or equivalent path) to identify specific bottlenecks (e.g., unbounded concurrency, inefficient string concatenation).
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Analysis and Reproduction' (Protocol in workflow.md)

## Phase 2: Refactoring
- [ ] Task: Refactor `read_many_files` to use a concurrency limit (e.g., using `p-limit` or a custom queue) for file system operations to prevent "EMFILE: too many open files" errors and reduce memory pressure.
- [ ] Task: Implement a check to strictly respect the `max_lines` or `limit` parameters per file early in the read process to avoid reading full content of massive files unnecessarily.
- [ ] Task: Optimize the result aggregation logic to handle large strings efficiently.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Refactoring' (Protocol in workflow.md)

## Phase 3: Verification and Cleanup
- [ ] Task: Run the benchmark script again against the new implementation and document the performance improvements (Time and Memory).
- [ ] Task: Run existing integration tests (`integration-tests/read_many_files.test.ts`) to ensure no regressions.
- [ ] Task: Add a new integration test case that specifically tests the tool with a larger-than-average set of files (e.g., 50-100 files) to ensure stability in CI.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Verification and Cleanup' (Protocol in workflow.md)
