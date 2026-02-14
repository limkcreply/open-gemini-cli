/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export interface LLMProviderConfig {
  name: string;
  baseURL: string;
  endpoint: string;
  headers: Record<string, string>;
  format: "openai" | "claude";
  streaming: boolean;
  defaultModel?: string;
  maxInputTokens?: number;
  maxOutputTokens?: number;
  requiresWebSearch?: boolean;
}

interface ProvidersConfig {
  providers: Record<string, LLMProviderConfig>;
}

/**
 * Substitute environment variables in a string
 * Example: "${OPENAI_API_KEY}" -> "sk-..."
 */
function substituteEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (match, varName) => {
    const envValue = process.env[varName];
    if (!envValue) {
      console.warn(
        `Warning: Environment variable ${varName} not set, using empty string`,
      );
      return "";
    }
    return envValue;
  });
}

/**
 * Recursively substitute environment variables in an object
 */
function substituteEnvVarsInObject(obj: any): any {
  if (typeof obj === "string") {
    return substituteEnvVars(obj);
  }
  if (Array.isArray(obj)) {
    return obj.map((item) => substituteEnvVarsInObject(item));
  }
  if (obj !== null && typeof obj === "object") {
    const result: any = {};
    for (const key in obj) {
      result[key] = substituteEnvVarsInObject(obj[key]);
    }
    return result;
  }
  return obj;
}

/**
 * Load LLM provider configuration
 * @param providerName - Name of the provider (e.g., "openai", "claude", "local-mlx")
 * @returns Provider configuration with environment variables substituted
 */
export function loadProviderConfig(providerName?: string): LLMProviderConfig {
  // Default to local-mlx if not specified
  const provider = providerName || process.env["LLM_PROVIDER"] || "local-mlx";

  // Load providers.json
  const configPath = path.join(__dirname, "llmProviders.json");
  const configData = fs.readFileSync(configPath, "utf-8");
  const config: ProvidersConfig = JSON.parse(configData);

  // Get provider config
  const providerConfig = config.providers[provider];
  if (!providerConfig) {
    throw new Error(
      `Unknown LLM provider: ${provider}. Available providers: ${Object.keys(config.providers).join(", ")}`,
    );
  }

  // Substitute environment variables
  const substitutedConfig = substituteEnvVarsInObject(
    providerConfig,
  ) as LLMProviderConfig;

  // Validate required fields
  if (!substitutedConfig.baseURL) {
    throw new Error(
      `Provider ${provider} has empty baseURL. Check environment variables.`,
    );
  }

  console.log(`📡 Using LLM provider: ${substitutedConfig.name} (${provider})`);
  console.log(
    `📡 Endpoint: ${substitutedConfig.baseURL}${substitutedConfig.endpoint}`,
  );

  return substitutedConfig;
}

/**
 * List all available providers
 */
export function listProviders(): string[] {
  const configPath = path.join(__dirname, "llmProviders.json");
  const configData = fs.readFileSync(configPath, "utf-8");
  const config: ProvidersConfig = JSON.parse(configData);
  return Object.keys(config.providers);
}
