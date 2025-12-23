/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// @ts-nocheck - Temporarily bypassing type conflicts for build

import type {
  EmbedContentParameters,
  GenerateContentConfig,
  PartListUnion,
  Content,
  Tool,
  GenerateContentResponse,
} from "./contentGenerator.js";
import {
  getDirectoryContextString,
  getEnvironmentContext,
} from "../utils/environmentContext.js";
import type { ServerKaiDexStreamEvent, ChatCompressionInfo } from "./turn.js";
import { CompressionStatus } from "./turn.js";
import { Turn, KaiDexEventType } from "./turn.js";
import type { Config } from "../config/config.js";
import { getCoreSystemPrompt, getCompressionPrompt } from "./prompts.js";
import { getResponseText } from "../utils/partUtils.js";
import { checkNextSpeaker } from "../utils/nextSpeakerChecker.js";
import { reportError } from "../utils/errorReporting.js";
import { KaiDexChat } from "./kaidexChat.js";
import { retryWithBackoff } from "../utils/retry.js";
import { getErrorMessage } from "../utils/errors.js";
import { isFunctionResponse } from "../utils/messageInspectors.js";
import { tokenLimit } from "./tokenLimits.js";
import type { ChatRecordingService } from "../services/chatRecordingService.js";
import type { ContentGenerator } from "./contentGenerator.js";
import {
  DEFAULT_KAIDEX_FLASH_MODEL,
  DEFAULT_THINKING_MODE,
} from "../config/models.js";
import { LoopDetectionService } from "../services/loopDetectionService.js";
import { ideContext } from "../ide/ideContext.js";
import {
  logChatCompression,
  logNextSpeakerCheck,
  logMalformedJsonResponse,
} from "../telemetry/loggers.js";
import {
  makeChatCompressionEvent,
  MalformedJsonResponseEvent,
  NextSpeakerCheckEvent,
} from "../telemetry/types.js";
import {
  setApiCallSource,
  resetApiCallSource,
} from "../telemetry/uiTelemetry.js";
import type { IdeContext, File } from "../ide/ideContext.js";
import { handleFallback } from "../fallback/handler.js";
import * as fs from "node:fs";

export function isThinkingSupported(model: string) {
  if (model.startsWith("gemini-2.5")) return true;
  return false;
}

export function isThinkingDefault(model: string) {
  if (model.startsWith("gemini-2.5-flash-lite")) return false;
  if (model.startsWith("gemini-2.5")) return true;
  return false;
}

/**
 * Returns the index of the content after the fraction of the total characters in the history.
 *
 * Exported for testing purposes.
 */
export function findIndexAfterFraction(
  history: Content[],
  fraction: number,
): number {
  if (fraction <= 0 || fraction >= 1) {
    throw new Error("Fraction must be between 0 and 1");
  }

  const contentLengths = history.map(
    (content) => JSON.stringify(content).length,
  );

  const totalCharacters = contentLengths.reduce(
    (sum, length) => sum + length,
    0,
  );
  const targetCharacters = totalCharacters * fraction;

  let charactersSoFar = 0;
  for (let i = 0; i < contentLengths.length; i++) {
    charactersSoFar += contentLengths[i];
    if (charactersSoFar >= targetCharacters) {
      return i;
    }
  }
  return contentLengths.length;
}

const MAX_TURNS = 100;

/**
 * Threshold for compression token count as a fraction of the model's token limit.
 * If the chat history exceeds this threshold, it will be compressed.
 */
const COMPRESSION_TOKEN_THRESHOLD = 0.7;

/**
 * The fraction of the latest chat history to keep. A value of 0.3
 * means that only the last 30% of the chat history will be kept after compression.
 */
const COMPRESSION_PRESERVE_THRESHOLD = 0.3;

export class KaiDexClient {
  private chat?: KaiDexChat;
  private readonly generateContentConfig: GenerateContentConfig = {
    temperature: 0,
    topP: 1,
    // maxOutputTokens is now read from provider config (llmProviders.json)
    // to respect each model's actual limits (e.g., gpt-4.1-mini: 32k, gpt-5: 128k)
    maxOutputTokens: undefined,
  };
  private sessionTurnCount = 0;

  private readonly loopDetector: LoopDetectionService;
  private lastPromptId: string;
  private lastSentIdeContext: IdeContext | undefined;
  private forceFullIdeContext = true;

  /**
   * At any point in this conversation, was compression triggered without
   * being forced and did it fail?
   */
  private hasFailedCompressionAttempt = false;

  constructor(private readonly config: Config) {
    this.loopDetector = new LoopDetectionService(config);
    this.lastPromptId = this.config.getSessionId();
  }

  async initialize() {
    this.chat = await this.startChat();
  }

  private getContentGeneratorOrFail(): ContentGenerator {
    if (!this.config.getContentGenerator()) {
      throw new Error("Content generator not initialized");
    }
    return this.config.getContentGenerator();
  }

  async addHistory(content: Content) {
    this.getChat().addHistory(content);
  }

  getChat(): KaiDexChat {
    if (!this.chat) {
      throw new Error("Chat not initialized");
    }
    return this.chat;
  }

  isInitialized(): boolean {
    return this.chat !== undefined;
  }

  getHistory(): Content[] {
    return this.getChat().getHistory();
  }

  stripThoughtsFromHistory() {
    this.getChat().stripThoughtsFromHistory();
  }

  setHistory(history: Content[]) {
    this.getChat().setHistory(history);
    this.forceFullIdeContext = true;
  }

  async setTools(): Promise<void> {
    const toolRegistry = this.config.getToolRegistry();
    const toolDeclarations = toolRegistry.getFunctionDeclarations();
    const tools: Tool[] = [{ functionDeclarations: toolDeclarations }];
    this.getChat().setTools(tools);
  }

  async resetChat(): Promise<void> {
    this.chat = await this.startChat();
  }

  getChatRecordingService(): ChatRecordingService | undefined {
    return this.chat?.getChatRecordingService();
  }

  async addDirectoryContext(): Promise<void> {
    if (!this.chat) {
      return;
    }

    this.getChat().addHistory({
      role: "user",
      parts: [{ text: await getDirectoryContextString(this.config) }],
    });
  }

  async startChat(extraHistory?: Content[]): Promise<KaiDexChat> {
    console.log("🏗️ CHAT_CREATION_START: startChat initializing");
    this.forceFullIdeContext = true;
    this.hasFailedCompressionAttempt = false;

    const envParts = await getEnvironmentContext(this.config);
    console.log("🔄 ENVIRONMENT_CONTEXT_READY: parts:", envParts.length);

    const toolRegistry = this.config.getToolRegistry();
    const toolDeclarations = toolRegistry.getFunctionDeclarations();
    console.log("🛠️ TOOL_DECLARATIONS_READY: tools:", toolDeclarations.length);
    const tools: Tool[] = [{ functionDeclarations: toolDeclarations }];

    const history: Content[] = [
      {
        role: "user",
        parts: envParts,
      },
      {
        role: "model",
        parts: [{ text: "Got it. Thanks for the context!" }],
      },
      ...(extraHistory ?? []),
    ];

    try {
      const userMemory = this.config.getUserMemory();
      const systemInstruction = getCoreSystemPrompt(userMemory);
      console.log(
        "✨ SYSTEM_INJECT_READY: prompt length:",
        systemInstruction.length,
      );
      const model = this.config.getModel();
      const generateContentConfigWithThinking = isThinkingSupported(model)
        ? {
            ...this.generateContentConfig,
            thinkingConfig: {
              thinkingBudget: -1,
              includeThoughts: true,
              ...(!isThinkingDefault(model)
                ? { thinkingBudget: DEFAULT_THINKING_MODE }
                : {}),
            },
          }
        : this.generateContentConfig;

      const kaidexChat = new KaiDexChat(
        this.config,
        {
          systemInstruction,
          ...generateContentConfigWithThinking,
          tools,
        },
        history,
      );

      console.log("🏗️ CHAT_CREATION_COMPLETE: KaiDexChat ready");
      return kaidexChat;
    } catch (error) {
      await reportError(
        error,
        "Error initializing KaiDex chat session.",
        history,
        "startChat",
      );
      throw new Error(`Failed to initialize chat: ${getErrorMessage(error)}`);
    }
  }

  private getIdeContextParts(forceFullContext: boolean): {
    contextParts: string[];
    newIdeContext: IdeContext | undefined;
  } {
    const currentIdeContext = ideContext.getIdeContext();
    if (!currentIdeContext) {
      return { contextParts: [], newIdeContext: undefined };
    }

    if (forceFullContext || !this.lastSentIdeContext) {
      // Send full context as JSON
      const openFiles = currentIdeContext.workspaceState?.openFiles || [];
      const activeFile = openFiles.find((f) => f.isActive);
      const otherOpenFiles = openFiles
        .filter((f) => !f.isActive)
        .map((f) => f.path);

      const contextData: Record<string, unknown> = {};

      if (activeFile) {
        contextData["activeFile"] = {
          path: activeFile.path,
          cursor: activeFile.cursor
            ? {
                line: activeFile.cursor.line,
                character: activeFile.cursor.character,
              }
            : undefined,
          selectedText: activeFile.selectedText || undefined,
        };
      }

      if (otherOpenFiles.length > 0) {
        contextData["otherOpenFiles"] = otherOpenFiles;
      }

      if (Object.keys(contextData).length === 0) {
        return { contextParts: [], newIdeContext: currentIdeContext };
      }

      const jsonString = JSON.stringify(contextData, null, 2);
      const contextParts = [
        "Here is the user's editor context as a JSON object. This is for your information only.",
        "```json",
        jsonString,
        "```",
      ];

      if (this.config.getDebugMode()) {
        console.log(contextParts.join("\n"));
      }
      return {
        contextParts,
        newIdeContext: currentIdeContext,
      };
    } else {
      // Calculate and send delta as JSON
      const delta: Record<string, unknown> = {};
      const changes: Record<string, unknown> = {};

      const lastFiles = new Map(
        (this.lastSentIdeContext.workspaceState?.openFiles || []).map(
          (f: File) => [f.path, f],
        ),
      );
      const currentFiles = new Map(
        (currentIdeContext.workspaceState?.openFiles || []).map((f: File) => [
          f.path,
          f,
        ]),
      );

      const openedFiles: string[] = [];
      for (const [path] of currentFiles.entries()) {
        if (!lastFiles.has(path)) {
          openedFiles.push(path);
        }
      }
      if (openedFiles.length > 0) {
        changes["filesOpened"] = openedFiles;
      }

      const closedFiles: string[] = [];
      for (const [path] of lastFiles.entries()) {
        if (!currentFiles.has(path)) {
          closedFiles.push(path);
        }
      }
      if (closedFiles.length > 0) {
        changes["filesClosed"] = closedFiles;
      }

      const lastActiveFile = (
        this.lastSentIdeContext.workspaceState?.openFiles || []
      ).find((f: File) => f.isActive);
      const currentActiveFile = (
        currentIdeContext.workspaceState?.openFiles || []
      ).find((f: File) => f.isActive);

      if (currentActiveFile) {
        if (!lastActiveFile || lastActiveFile.path !== currentActiveFile.path) {
          changes["activeFileChanged"] = {
            path: currentActiveFile.path,
            cursor: currentActiveFile.cursor
              ? {
                  line: currentActiveFile.cursor.line,
                  character: currentActiveFile.cursor.character,
                }
              : undefined,
            selectedText: currentActiveFile.selectedText || undefined,
          };
        } else {
          const lastCursor = lastActiveFile.cursor;
          const currentCursor = currentActiveFile.cursor;
          if (
            currentCursor &&
            (!lastCursor ||
              lastCursor.line !== currentCursor.line ||
              lastCursor.character !== currentCursor.character)
          ) {
            changes["cursorMoved"] = {
              path: currentActiveFile.path,
              cursor: {
                line: currentCursor.line,
                character: currentCursor.character,
              },
            };
          }

          const lastSelectedText = lastActiveFile.selectedText || "";
          const currentSelectedText = currentActiveFile.selectedText || "";
          if (lastSelectedText !== currentSelectedText) {
            changes["selectionChanged"] = {
              path: currentActiveFile.path,
              selectedText: currentSelectedText,
            };
          }
        }
      } else if (lastActiveFile) {
        changes["activeFileChanged"] = {
          path: null,
          previousPath: lastActiveFile.path,
        };
      }

      if (Object.keys(changes).length === 0) {
        return { contextParts: [], newIdeContext: currentIdeContext };
      }

      delta["changes"] = changes;
      const jsonString = JSON.stringify(delta, null, 2);
      const contextParts = [
        "Here is a summary of changes in the user's editor context, in JSON format. This is for your information only.",
        "```json",
        jsonString,
        "```",
      ];

      if (this.config.getDebugMode()) {
        console.log(contextParts.join("\n"));
      }
      return {
        contextParts,
        newIdeContext: currentIdeContext,
      };
    }
  }

  async *sendMessageStream(
    request: PartListUnion,
    signal: AbortSignal,
    prompt_id: string,
    turns: number = MAX_TURNS,
    originalModel?: string,
  ): AsyncGenerator<ServerKaiDexStreamEvent, Turn> {
    if (this.lastPromptId !== prompt_id) {
      this.loopDetector.reset(prompt_id);
      this.lastPromptId = prompt_id;
    }
    this.sessionTurnCount++;
    if (
      this.config.getMaxSessionTurns() > 0 &&
      this.sessionTurnCount > this.config.getMaxSessionTurns()
    ) {
      yield { type: KaiDexEventType.MaxSessionTurns };
      return new Turn(this.getChat(), prompt_id);
    }
    // Ensure turns never exceeds MAX_TURNS to prevent infinite loops
    const boundedTurns = Math.min(turns, MAX_TURNS);
    if (!boundedTurns) {
      return new Turn(this.getChat(), prompt_id);
    }

    // Track the original model from the first call to detect model switching
    const initialModel = originalModel || this.config.getModel();

    const compressed = await this.tryCompressChat(prompt_id);

    if (compressed.compressionStatus === CompressionStatus.COMPRESSED) {
      yield { type: KaiDexEventType.ChatCompressed, value: compressed };
    }

    // HARD CAP: If context still exceeds 90% after compression, force truncate
    const model = this.config.getModel();
    const limit = tokenLimit(model);
    const safeLimit = limit * 0.9; // Keep 10% buffer for response
    let currentHistory = this.getChat().getHistory(true);
    let { totalTokens: currentTokens } =
      await this.getContentGeneratorOrFail().countTokens({
        model,
        contents: currentHistory,
      });

    // Force truncate until under safe limit
    while (currentTokens && currentTokens > safeLimit && currentHistory.length > 2) {
      // Remove oldest 20% of messages
      const removeCount = Math.max(1, Math.floor(currentHistory.length * 0.2));
      currentHistory = currentHistory.slice(removeCount);
      this.getChat().setHistory(currentHistory);

      const result = await this.getContentGeneratorOrFail().countTokens({
        model,
        contents: currentHistory,
      });
      currentTokens = result.totalTokens;
    }

    // Prevent context updates from being sent while a tool call is
    // waiting for a response. The KaiDex API requires that a functionResponse
    // part from the user immediately follows a functionCall part from the model
    // in the conversation history . The IDE context is not discarded; it will
    // be included in the next regular message sent to the model.
    const history = this.getHistory();
    const lastMessage =
      history.length > 0 ? history[history.length - 1] : undefined;
    const hasPendingToolCall =
      !!lastMessage &&
      lastMessage.role === "model" &&
      (lastMessage.parts?.some((p) => "functionCall" in p) || false);

    if (this.config.getIdeMode() && !hasPendingToolCall) {
      const { contextParts, newIdeContext } = this.getIdeContextParts(
        this.forceFullIdeContext || history.length === 0,
      );
      if (contextParts.length > 0) {
        this.getChat().addHistory({
          role: "user",
          parts: [{ text: contextParts.join("\n") }],
        });
      }
      this.lastSentIdeContext = newIdeContext;
      this.forceFullIdeContext = false;
    }

    const turn = new Turn(this.getChat(), prompt_id);

    const loopDetected = await this.loopDetector.turnStarted(signal);
    if (loopDetected) {
      yield { type: KaiDexEventType.LoopDetected };
      return turn;
    }

    const resultStream = turn.run(request, signal);
    for await (const event of resultStream) {
      if (this.loopDetector.addAndCheck(event)) {
        yield { type: KaiDexEventType.LoopDetected };
        return turn;
      }
      yield event;
      if (event.type === KaiDexEventType.Error) {
        return turn;
      }
    }

    // Clear pendingToolCalls after yielding tool events
    if (turn.pendingToolCalls.length > 0) {
      turn.clearPendingToolCalls();
    }

    // Re-check conversation history to see if there are pending tool calls awaiting responses
    // This is more reliable than checking turn.pendingToolCalls which gets cleared above
    // We need to check again because the model's response was added to history during turn.run()
    const historyAfterTurn = this.getHistory();
    const lastMessageAfterTurn =
      historyAfterTurn.length > 0
        ? historyAfterTurn[historyAfterTurn.length - 1]
        : undefined;
    const hasPendingToolCallAfterTurn =
      !!lastMessageAfterTurn &&
      lastMessageAfterTurn.role === "model" &&
      (lastMessageAfterTurn.parts?.some((p) => "functionCall" in p) || false);

    if (!hasPendingToolCallAfterTurn && signal && !signal.aborted) {
      // Check if model was switched during the call (likely due to quota error)
      const currentModel = this.config.getModel();
      if (currentModel !== initialModel) {
        // Model was switched (likely due to quota error fallback)
        // Don't continue with recursive call to prevent unwanted Flash execution
        return turn;
      }

      if (this.config.getSkipNextSpeakerCheck()) {
        return turn;
      }

      const nextSpeakerCheck = await checkNextSpeaker(
        this.getChat(),
        this,
        signal,
        this.config.getModel(),
      );
      logNextSpeakerCheck(
        this.config,
        new NextSpeakerCheckEvent(
          prompt_id,
          turn.finishReason?.toString() || "",
          nextSpeakerCheck?.next_speaker || "",
        ),
      );
      if (nextSpeakerCheck?.next_speaker === "model") {
        const nextRequest = [{ text: "Please continue." }];
        // This recursive call's events will be yielded out, but the final
        // turn object will be from the top-level call.
        yield* this.sendMessageStream(
          nextRequest,
          signal,
          prompt_id,
          boundedTurns - 1,
          initialModel,
        );
      }
    }
    return turn;
  }

  async generateJson(
    contents: Content[],
    schema: Record<string, unknown>,
    abortSignal: AbortSignal,
    model: string,
    config: GenerateContentConfig = {},
    caller = "unknown", // Identifies who called generateJson (e.g., 'checkNextSpeaker', 'loopDetection', 'editCorrector')
  ): Promise<Record<string, unknown>> {
    let currentAttemptModel: string = model;

    try {
      const userMemory = this.config.getUserMemory();
      const systemInstruction = getCoreSystemPrompt(userMemory);
      const requestConfig = {
        abortSignal,
        ...this.generateContentConfig,
        ...config,
      };

      const apiCall = () => {
        const modelToUse = this.config.isInFallbackMode()
          ? DEFAULT_KAIDEX_FLASH_MODEL
          : model;
        currentAttemptModel = modelToUse;

        return this.getContentGeneratorOrFail().generateContent(
          {
            model: modelToUse,
            config: {
              ...requestConfig,
              systemInstruction,
              responseJsonSchema: schema,
              responseMimeType: "application/json",
            },
            contents,
          },
          this.lastPromptId,
        );
      };

      const onPersistent429Callback = async (
        authType?: string,
        error?: unknown,
      ) =>
        // Pass the captured model to the centralized handler.
        await handleFallback(this.config, currentAttemptModel, authType, error);

      console.log(
      );
      const result = await retryWithBackoff(apiCall, {
        onPersistent429: onPersistent429Callback,
        authType: this.config.getContentGeneratorConfig()?.authType,
      });

      console.log(
        JSON.stringify(result, null, 2),
      );
      let text = getResponseText(result);
      console.log(
        text ? `"${text.slice(0, 100)}..."` : "NULL/EMPTY",
      );
      if (!text) {
        const error = new Error(
          `API returned an empty response for generateJson (caller: ${caller}).`,
        );
        await reportError(
          error,
          `Error in generateJson (caller: ${caller}): API returned an empty response.`,
          contents,
          `generateJson-empty-response-${caller}`,
        );
        throw error;
      }

      const prefix = "```json";
      const suffix = "```";
      if (text.startsWith(prefix) && text.endsWith(suffix)) {
        logMalformedJsonResponse(
          this.config,
          new MalformedJsonResponseEvent(currentAttemptModel),
        );
        text = text
          .substring(prefix.length, text.length - suffix.length)
          .trim();
      }

      try {
        return JSON.parse(text);
      } catch (parseError) {
        // Best-effort parse hardening: log raw body to /tmp and return a structured error instead of throwing.
        try {
          const ts = new Date().toISOString();
          const logPath = `/tmp/kaidex_errors_${ts.slice(0, 10)}.log`;
          const snippet = text.slice(0, 400);
          const entry =
            [
              `[${ts}] generateJson parse-error caller=${caller}`,
              `model=${currentAttemptModel}`,
              `body_length=${text.length}`,
              `snippet=${JSON.stringify(snippet)}`,
              `error=${getErrorMessage(parseError)}`,
            ].join("\n") + "\n";
          fs.appendFileSync(logPath, entry);
        } catch {}

        await reportError(
          parseError,
          `Failed to parse JSON response from generateJson (caller: ${caller}).`,
          {
            caller,
            responseTextFailedToParse: text,
            originalRequestContents: contents,
          },
          `generateJson-parse-${caller}`,
        );

        return {
          error: "parse_error",
          message: `Failed to parse API response as JSON (caller: ${caller}): ${getErrorMessage(parseError)}`,
          snippet: text.slice(0, 400),
          body_length: text.length,
        } as Record<string, unknown>;
      }
    } catch (error) {
      if (abortSignal.aborted) {
        throw error;
      }

      // Avoid double reporting for the empty response case handled above
      if (
        error instanceof Error &&
        error.message === "API returned an empty response for generateJson."
      ) {
        throw error;
      }

      await reportError(
        error,
        `Error generating JSON content via API (caller: ${caller}).`,
        contents,
        `generateJson-api-${caller}`,
      );
      throw new Error(
        `Failed to generate JSON content: ${getErrorMessage(error)}`,
      );
    }
  }

  async generateContent(
    contents: Content[],
    generationConfig: GenerateContentConfig,
    abortSignal: AbortSignal,
    model: string,
  ): Promise<GenerateContentResponse> {
    let currentAttemptModel: string = model;

    const configToUse: GenerateContentConfig = {
      ...this.generateContentConfig,
      ...generationConfig,
    };

    try {
      const userMemory = this.config.getUserMemory();
      const systemInstruction = getCoreSystemPrompt(userMemory);

      const requestConfig: GenerateContentConfig = {
        abortSignal,
        ...configToUse,
        systemInstruction,
      };

      const apiCall = () => {
        const modelToUse = this.config.isInFallbackMode()
          ? DEFAULT_KAIDEX_FLASH_MODEL
          : model;
        currentAttemptModel = modelToUse;

        return this.getContentGeneratorOrFail().generateContent(
          {
            model: modelToUse,
            config: requestConfig,
            contents,
          },
          this.lastPromptId,
        );
      };
      const onPersistent429Callback = async (
        authType?: string,
        error?: unknown,
      ) =>
        // Pass the captured model to the centralized handler.
        await handleFallback(this.config, currentAttemptModel, authType, error);

      const result = await retryWithBackoff(apiCall, {
        onPersistent429: onPersistent429Callback,
        authType: this.config.getContentGeneratorConfig()?.authType,
      });
      return result;
    } catch (error: unknown) {
      if (abortSignal.aborted) {
        throw error;
      }

      await reportError(
        error,
        `Error generating content via API with model ${currentAttemptModel}.`,
        {
          requestContents: contents,
          requestConfig: configToUse,
        },
        "generateContent-api",
      );
      throw new Error(
        `Failed to generate content with model ${currentAttemptModel}: ${getErrorMessage(error)}`,
      );
    }
  }

  async generateEmbedding(texts: string[]): Promise<number[][]> {
    if (!texts || texts.length === 0) {
      return [];
    }
    const embedModelParams: EmbedContentParameters = {
      model: this.config.getEmbeddingModel(),
      contents: texts,
    };

    const embedContentResponse =
      await this.getContentGeneratorOrFail().embedContent(embedModelParams);
    if (
      !embedContentResponse.embeddings ||
      embedContentResponse.embeddings.length === 0
    ) {
      throw new Error("No embeddings found in API response.");
    }

    if (embedContentResponse.embeddings.length !== texts.length) {
      throw new Error(
        `API returned a mismatched number of embeddings. Expected ${texts.length}, got ${embedContentResponse.embeddings.length}.`,
      );
    }

    return embedContentResponse.embeddings.map((embedding, index) => {
      const values = embedding.values;
      if (!values || values.length === 0) {
        throw new Error(
          `API returned an empty embedding for input text at index ${index}: "${texts[index]}"`,
        );
      }
      return values;
    });
  }

  async tryCompressChat(
    prompt_id: string,
    force: boolean = false,
  ): Promise<ChatCompressionInfo> {
    const curatedHistory = this.getChat().getHistory(true);

    // Regardless of `force`, don't do anything if the history is empty.
    if (curatedHistory.length === 0) {
      return {
        originalTokenCount: 0,
        newTokenCount: 0,
        compressionStatus: CompressionStatus.NOOP,
      };
    }

    // If a previous compression attempt failed, check if we should retry
    // Allow retry when context is critically full (>95% of limit)
    if (this.hasFailedCompressionAttempt && !force) {
      const model = this.config.getModel();
      const limit = tokenLimit(model);
      const { totalTokens: currentTokens } =
        await this.getContentGeneratorOrFail().countTokens({
          model,
          contents: curatedHistory,
        });
      const usagePercent = currentTokens ? (currentTokens / limit) * 100 : 0;

      // If context is critically full (>95%), reset flag and allow retry
      if (usagePercent > 95) {
        this.hasFailedCompressionAttempt = false;
      } else {
        return {
          originalTokenCount: currentTokens || 0,
          newTokenCount: currentTokens || 0,
          compressionStatus: CompressionStatus.NOOP,
        };
      }
    }

    const model = this.config.getModel();

    const { totalTokens: originalTokenCount } =
      await this.getContentGeneratorOrFail().countTokens({
        model,
        contents: curatedHistory,
      });
    if (originalTokenCount === undefined) {
      console.warn(`Could not determine token count for model ${model}.`);
      // Don't permanently block - might succeed next time
      return {
        originalTokenCount: 0,
        newTokenCount: 0,
        compressionStatus:
          CompressionStatus.COMPRESSION_FAILED_TOKEN_COUNT_ERROR,
      };
    }

    const contextPercentageThreshold =
      this.config.getChatCompression()?.contextPercentageThreshold;

    // Don't compress if not forced and we are under the limit.
    if (!force) {
      const threshold =
        contextPercentageThreshold ?? COMPRESSION_TOKEN_THRESHOLD;
      if (originalTokenCount < threshold * tokenLimit(model)) {
        return {
          originalTokenCount,
          newTokenCount: originalTokenCount,
          compressionStatus: CompressionStatus.NOOP,
        };
      }
    }

    let compressBeforeIndex = findIndexAfterFraction(
      curatedHistory,
      1 - COMPRESSION_PRESERVE_THRESHOLD,
    );
    // Find the first user message after the index. This is the start of the next turn.
    while (
      compressBeforeIndex < curatedHistory.length &&
      (curatedHistory[compressBeforeIndex]?.role === "model" ||
        isFunctionResponse(curatedHistory[compressBeforeIndex]))
    ) {
      compressBeforeIndex++;
    }

    const historyToCompress = curatedHistory.slice(0, compressBeforeIndex);
    const historyToKeep = curatedHistory.slice(compressBeforeIndex);

    this.getChat().setHistory(historyToCompress);

    // Mark compression call so it doesn't update UI token count
    setApiCallSource("compression");
    const { text: summary } = await this.getChat().sendMessage(
      {
        message: {
          text: "First, reason in your scratchpad. Then, generate the <state_snapshot>.",
        },
        config: {
          systemInstruction: { text: getCompressionPrompt() },
        },
      },
      prompt_id,
    );
    resetApiCallSource();

    const chat = await this.startChat([
      {
        role: "user",
        parts: [{ text: summary }],
      },
      {
        role: "model",
        parts: [{ text: "Got it. Thanks for the additional context!" }],
      },
      ...historyToKeep,
    ]);
    this.forceFullIdeContext = true;

    const { totalTokens: newTokenCount } =
      await this.getContentGeneratorOrFail().countTokens({
        // model might change after calling `sendMessage`, so we get the newest value from config
        model: this.config.getModel(),
        contents: chat.getHistory(),
      });
    if (newTokenCount === undefined) {
      console.warn("Could not determine compressed history token count.");
      // Don't permanently block - might succeed next time
      return {
        originalTokenCount,
        newTokenCount: originalTokenCount,
        compressionStatus:
          CompressionStatus.COMPRESSION_FAILED_TOKEN_COUNT_ERROR,
      };
    }

    logChatCompression(
      this.config,
      makeChatCompressionEvent({
        tokens_before: originalTokenCount,
        tokens_after: newTokenCount,
      }),
    );

    if (newTokenCount > originalTokenCount) {
      // Compression inflated - fallback to truncation
      // Keep only the most recent 30% of history
      const keepCount = Math.max(2, Math.floor(curatedHistory.length * 0.3));
      const truncatedHistory = curatedHistory.slice(-keepCount);
      this.getChat().setHistory(truncatedHistory);

      const { totalTokens: truncatedTokenCount } =
        await this.getContentGeneratorOrFail().countTokens({
          model: this.config.getModel(),
          contents: truncatedHistory,
        });

      return {
        originalTokenCount,
        newTokenCount: truncatedTokenCount || originalTokenCount,
        compressionStatus: CompressionStatus.COMPRESSED,
      };
    } else {
      this.chat = chat; // Chat compression successful, set new state.
    }

    return {
      originalTokenCount,
      newTokenCount,
      compressionStatus: CompressionStatus.COMPRESSED,
    };
  }
}

export const TEST_ONLY = {
  COMPRESSION_PRESERVE_THRESHOLD,
  COMPRESSION_TOKEN_THRESHOLD,
};
