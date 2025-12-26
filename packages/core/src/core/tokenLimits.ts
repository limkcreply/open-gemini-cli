/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from "node:fs";
import * as path from "node:path";

type Model = string;
type TokenCount = number;

type ProvidersConfig = {
  providers: Record<
    string,
    {
      defaultModel?: string;
      maxInputTokens?: number;
    }
  >;
};

// Token limit constants for different model types
export const QWEN3_CODER_30B = 256_000; // Qwen3-Coder-30B context limit
export const GOOGLE_GEMINI_2_5 = 1_048_576; // Standard Gemini 2.5 models
export const GOOGLE_GEMINI_1_5_PRO = 2_097_152; // Gemini 1.5 Pro

// Default for local LLMs and unknown models
export const DEFAULT_TOKEN_LIMIT = 256_000;

// globalThis.__dirname is set by esbuild banner - points to bundle directory
const bundleDirname = (globalThis as any).__dirname as string;

let providerConfigCache: ProvidersConfig | null = null;
let providerConfigLoadFailed = false;

function loadProvidersConfig(): ProvidersConfig | null {
  if (providerConfigCache || providerConfigLoadFailed) {
    return providerConfigCache;
  }

  // llmProviders.json is sibling to kaidex.js in bundle/
  const configPath = path.join(bundleDirname, "llmProviders.json");

  try {
    const raw = fs.readFileSync(configPath, "utf-8");
    providerConfigCache = JSON.parse(raw) as ProvidersConfig;
    return providerConfigCache;
  } catch (err) {
    providerConfigLoadFailed = true;
    console.warn(
      `tokenLimit: unable to read llmProviders.json at ${configPath}: ${String(err)}`,
    );
    return null;
  }
}

function resolveProviderTokenLimit(model: Model): TokenCount | undefined {
  const config = loadProvidersConfig();
  if (!config) return undefined;

  for (const [providerKey, provider] of Object.entries(config.providers)) {
    if (provider.defaultModel === model || providerKey === model) {
      if (provider.maxInputTokens !== undefined) {
        return provider.maxInputTokens;
      }
      console.warn(
        `tokenLimit: provider "${providerKey}" matched model "${model}" but has no maxInputTokens; using fallback`,
      );
      return undefined;
    }
  }

  return undefined;
}

export function tokenLimit(model: Model): TokenCount {
  const providerLimit = resolveProviderTokenLimit(model);
  if (providerLimit !== undefined) {
    return providerLimit;
  }

  // Add other models as they become relevant or if specified by config
  // Pulled from https://ai.google.dev/gemini-api/docs/models
  switch (model) {
    case "gemini-1.5-pro":
      return GOOGLE_GEMINI_1_5_PRO;
    case "gemini-1.5-flash":
    case "gemini-2.5-pro-preview-05-06":
    case "gemini-2.5-pro-preview-06-05":
    case "gemini-2.5-pro":
    case "gemini-2.5-flash-preview-05-20":
    case "gemini-2.5-flash":
    case "gemini-2.5-flash-lite":
    case "gemini-2.0-flash":
    case "gemini-3-pro-preview":
      return GOOGLE_GEMINI_2_5;
    case "gemini-2.0-flash-preview-image-generation":
      return 32_000;
    // Local LLM models
    case "kaidex-server":
    case "qwen32b":
    case "gemma3-27b":
    case "magistral24b":
      return QWEN3_CODER_30B;
    default:
      return DEFAULT_TOKEN_LIMIT;
  }
}
