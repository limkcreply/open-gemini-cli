# Specification: Refactor `read_many_files` Tool

## 1. Overview
The `read_many_files` tool is a critical utility for the agent to ingest context from multiple files. Currently, it may face performance bottlenecks or high memory usage when processing a very large number of files (e.g., thousands of files in a large repo scan) or large individual files. This track aims to refactor the tool to handle such cases efficiently.

## 2. Goals
- **Efficiency:** Reduce memory footprint during large file operations.
- **Robustness:** Prevent crashes or timeouts when handling large datasets.
- **Maintainability:** Ensure code is clean and well-tested.

## 3. Requirements
- **Functionality:** The tool must still support glob patterns, exclusion lists, and correct file reading.
- **Performance:**
    - Should implement batching or streaming if possible to avoid loading everything into memory at once if not necessary (though the final output is a concatenated string, intermediate processing should be efficient).
    - Consider implementing a hard limit or pagination if the result exceeds context window limits (though this might be a separate token limit concern, efficient reading is the first step).
- **Output:** The output format (concatenated string with separators) must remain consistent for backward compatibility.

## 4. Proposed Changes
- Analyze current implementation for bottlenecks.
- Introduce concurrency limits for file reading operations (e.g., `p-limit` or similar pattern) to avoid opening too many file descriptors simultaneously.
- Optimize the concatenation process.
- Add safeguards for extremely large files (skip or truncate with warning).

## 5. Verification Plan
- Create a test case with a large number of dummy files (e.g., 1000 small files).
- Benchmark the time and memory usage before and after refactoring.
- Ensure all existing integration tests pass.
