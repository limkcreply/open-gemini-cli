/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  Content,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  GenerateContentParameters,
  GenerateContentResponse,
} from "./contentGenerator.js";

// Usage metadata interface for logging
export interface GenerateContentResponseUsageMetadata {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  totalTokenCount?: number;
  cachedContentTokenCount?: number;
  thoughtsTokenCount?: number;
  toolUsePromptTokenCount?: number;
}
import {
  ApiRequestEvent,
  ApiResponseEvent,
  ApiErrorEvent,
} from "../telemetry/types.js";
import type { Config } from "../config/config.js";
import {
  logApiError,
  logApiRequest,
  logApiResponse,
} from "../telemetry/loggers.js";
import type { ContentGenerator } from "./contentGenerator.js";
import { isStructuredError } from "../utils/quotaErrorDetection.js";

interface StructuredError {
  status: number;
}

/**
 * A decorator that wraps a ContentGenerator to add logging to API calls.
 */
export class LoggingContentGenerator implements ContentGenerator {
  constructor(
    private readonly wrapped: ContentGenerator,
    private readonly config: Config,
  ) {}

  getWrapped(): ContentGenerator {
    return this.wrapped;
  }

  private logApiRequest(
    contents: Content[],
    model: string,
    promptId: string,
  ): void {
    const requestText = JSON.stringify(contents);
    logApiRequest(
      this.config,
      new ApiRequestEvent(model, promptId, requestText),
    );
  }

  private _logApiResponse(
    durationMs: number,
    model: string,
    prompt_id: string,
    usageMetadata?: GenerateContentResponseUsageMetadata,
    responseText?: string,
  ): void {
    logApiResponse(
      this.config,
      new ApiResponseEvent(
        model,
        durationMs,
        prompt_id,
        this.config.getContentGeneratorConfig()?.authType,
        usageMetadata,
        responseText,
      ),
    );
  }

  private _logApiError(
    durationMs: number,
    error: unknown,
    model: string,
    prompt_id: string,
  ): void {
    const errorMessage = error instanceof Error ? error.message : String(error);
    const errorType = error instanceof Error ? error.name : "unknown";

    logApiError(
      this.config,
      new ApiErrorEvent(
        model,
        errorMessage,
        durationMs,
        prompt_id,
        this.config.getContentGeneratorConfig()?.authType,
        errorType,
        isStructuredError(error)
          ? (error as StructuredError).status
          : undefined,
      ),
    );
  }

  async generateContent(
    req: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    const startTime = Date.now();
    const modelName = req.model || "unknown";
    this.logApiRequest(req.contents, modelName, userPromptId);
    try {
      const response = await this.wrapped.generateContent(req, userPromptId);
      const durationMs = Date.now() - startTime;
      this._logApiResponse(
        durationMs,
        modelName,
        userPromptId,
        (response as any).usageMetadata,
        JSON.stringify(response),
      );
      return response;
    } catch (error) {
      const durationMs = Date.now() - startTime;
      this._logApiError(durationMs, error, modelName, userPromptId);
      throw error;
    }
  }

  async *generateContentStream(
    req: GenerateContentParameters,
    userPromptId: string,
  ): AsyncGenerator<GenerateContentResponse, any, any> {
    const startTime = Date.now();
    const modelName = req.model || "unknown";
    this.logApiRequest(req.contents, modelName, userPromptId);

    let stream: AsyncGenerator<GenerateContentResponse>;
    try {
      // await handles both AsyncGenerator (OpenAI path) and Promise<AsyncGenerator> (native Gemini SDK)
      stream = await Promise.resolve(this.wrapped.generateContentStream(req, userPromptId));
    } catch (error) {
      const durationMs = Date.now() - startTime;
      this._logApiError(durationMs, error, modelName, userPromptId);
      throw error;
    }

    yield* this.loggingStreamWrapper(
      stream,
      startTime,
      userPromptId,
      modelName,
    );
  }

  private async *loggingStreamWrapper(
    stream: AsyncGenerator<GenerateContentResponse>,
    startTime: number,
    userPromptId: string,
    modelName: string,
  ): AsyncGenerator<GenerateContentResponse> {
    const responses: GenerateContentResponse[] = [];

    let lastUsageMetadata: GenerateContentResponseUsageMetadata | undefined;
    try {
      for await (const response of stream) {
        responses.push(response);
        if ((response as any).usageMetadata) {
          lastUsageMetadata = (response as any).usageMetadata;
        }
        yield response;
      }
      // Only log successful API response if no error occurred
      const durationMs = Date.now() - startTime;
      this._logApiResponse(
        durationMs,
        modelName,
        userPromptId,
        lastUsageMetadata,
        JSON.stringify(responses),
      );
    } catch (error) {
      const durationMs = Date.now() - startTime;
      this._logApiError(durationMs, error, modelName, userPromptId);
      throw error;
    }
  }

  async countTokens(req: CountTokensParameters): Promise<CountTokensResponse> {
    return this.wrapped.countTokens(req);
  }

  async embedContent(
    req: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    return this.wrapped.embedContent(req);
  }

  // OpenAI-compatible primary interface methods
  chatCompletion(request: any, userPromptId: string): Promise<any> {
    return this.wrapped.chatCompletion(request, userPromptId);
  }

  chatCompletionStream(
    request: any,
    userPromptId: string,
  ): AsyncGenerator<any> {
    return this.wrapped.chatCompletionStream(request, userPromptId);
  }
}
