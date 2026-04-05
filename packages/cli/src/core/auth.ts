/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  type AuthType,
  type Config,
  getErrorMessage,
  AuthType as AuthTypeEnum,
} from "@google/kaidex-cli-core";

/**
 * Handles the initial authentication flow.
 * @param config The application config.
 * @param authType The selected auth type from settings.
 * @returns An error message if authentication fails, otherwise null.
 */
export async function performInitialAuth(
  config: Config,
  authType: AuthType | undefined,
): Promise<string | null> {
  // BYPASS_AUTH is for local LLM usage — skip all auth
  if (process.env["BYPASS_AUTH"] === "true") {
    try {
      await config.refreshAuth(AuthTypeEnum.LOCAL_LLM);
    } catch (e) {
      return `Failed to initialize local LLM: ${getErrorMessage(e)}`;
    }
    return null;
  }

  // LLM_PROVIDER + env keys drive routing — always initialize content generator
  try {
    await config.refreshAuth(authType || AuthTypeEnum.USE_GEMINI);
  } catch (e) {
    return `Failed to initialize: ${getErrorMessage(e)}`;
  }

  return null;
}
