/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { copyFileSync, cpSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { glob } from "glob";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");
const bundleDir = join(root, "bundle");

// Create the bundle directory if it doesn't exist
if (!existsSync(bundleDir)) {
  mkdirSync(bundleDir);
}

// Find and copy all .sb files from packages to the root of the bundle directory
const sbFiles = glob.sync("packages/**/*.sb", { cwd: root });
for (const file of sbFiles) {
  copyFileSync(join(root, file), join(bundleDir, basename(file)));
}

// Copy llmProviders.json to bundle directory
const llmProvidersPath = join(
  root,
  "packages/core/src/config/llmProviders.json",
);
if (existsSync(llmProvidersPath)) {
  copyFileSync(llmProvidersPath, join(bundleDir, "llmProviders.json"));
}

// Copy VS Code extension .vsix to bundle directory
const vsixFiles = glob.sync("packages/vscode-ide-companion/*.vsix", {
  cwd: root,
});
for (const file of vsixFiles) {
  copyFileSync(join(root, file), join(bundleDir, basename(file)));
}

// Copy bundled extensions to bundle directory
const bundledExtensionsPath = join(
  root,
  "packages/cli/src/bundled-extensions",
);
if (existsSync(bundledExtensionsPath)) {
  const bundleExtensionsDir = join(bundleDir, "bundled-extensions");
  cpSync(bundledExtensionsPath, bundleExtensionsDir, { recursive: true });
}

console.log("Assets copied to bundle/");
