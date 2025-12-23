/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  Part,
  PartListUnion,
  GenerateContentResponse,
  FunctionCall,
  FunctionDeclaration,
  FinishReason,
} from "./contentGenerator.js";
import type { GenerateContentResponseUsageMetadata } from "./loggingContentGenerator.js";
import type {
  ToolCallConfirmationDetails,
  ToolResult,
  ToolResultDisplay,
} from "../tools/tools.js";
import type { ToolErrorType } from "../tools/tool-error.js";
import { getResponseText } from "../utils/partUtils.js";
import { reportError } from "../utils/errorReporting.js";
import {
  getErrorMessage,
  UnauthorizedError,
  toFriendlyError,
} from "../utils/errors.js";
import type { KaiDexChat } from "./kaidexChat.js";
import { createKaidexUserContent } from "./kaidexChat.js";

// Define a structure for tools passed to the server
export interface ServerTool {
  name: string;
  schema: FunctionDeclaration;
  // The execute method signature might differ slightly or be wrapped
  execute(
    params: Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<ToolResult>;
  shouldConfirmExecute(
    params: Record<string, unknown>,
    abortSignal: AbortSignal,
  ): Promise<ToolCallConfirmationDetails | false>;
}

export enum KaiDexEventType {
  Content = "content",
  ToolCallRequest = "tool_call_request",
  ToolCallResponse = "tool_call_response",
  ToolCallConfirmation = "tool_call_confirmation",
  UserCancelled = "user_cancelled",
  Error = "error",
  ChatCompressed = "chat_compressed",
  Thought = "thought",
  MaxSessionTurns = "max_session_turns",
  Finished = "finished",
  LoopDetected = "loop_detected",
  Citation = "citation",
  Retry = "retry",
}

export type ServerKaiDexRetryEvent = {
  type: KaiDexEventType.Retry;
};

export interface StructuredError {
  message: string;
  status?: number;
}

export interface KaiDexErrorEventValue {
  error: StructuredError;
}

export interface KaiDexFinishedEventValue {
  reason: FinishReason | undefined;
  usageMetadata: GenerateContentResponseUsageMetadata | undefined;
}

export interface ToolCallRequestInfo {
  callId: string;
  name: string;
  args: Record<string, unknown>;
  isClientInitiated: boolean;
  prompt_id: string;
  // Gemini 3 Pro: encrypted reasoning state that must be passed back with function responses
  thoughtSignature?: string;
}

export interface ToolCallResponseInfo {
  callId: string;
  responseParts: Part[];
  resultDisplay: ToolResultDisplay | undefined;
  error: Error | undefined;
  errorType: ToolErrorType | undefined;
  outputFile?: string | undefined;
}

export interface ServerToolCallConfirmationDetails {
  request: ToolCallRequestInfo;
  details: ToolCallConfirmationDetails;
}

export type ThoughtSummary = {
  subject: string;
  description: string;
};

export type ServerKaiDexContentEvent = {
  type: KaiDexEventType.Content;
  value: string;
};

export type ServerKaiDexThoughtEvent = {
  type: KaiDexEventType.Thought;
  value: ThoughtSummary;
};

export type ServerKaiDexToolCallRequestEvent = {
  type: KaiDexEventType.ToolCallRequest;
  value: ToolCallRequestInfo;
};

export type ServerKaiDexToolCallResponseEvent = {
  type: KaiDexEventType.ToolCallResponse;
  value: ToolCallResponseInfo;
};

export type ServerKaiDexToolCallConfirmationEvent = {
  type: KaiDexEventType.ToolCallConfirmation;
  value: ServerToolCallConfirmationDetails;
};

export type ServerKaiDexUserCancelledEvent = {
  type: KaiDexEventType.UserCancelled;
};

export type ServerKaiDexErrorEvent = {
  type: KaiDexEventType.Error;
  value: KaiDexErrorEventValue;
};

export enum CompressionStatus {
  /** The compression was successful */
  COMPRESSED = 1,

  /** The compression failed due to the compression inflating the token count */
  COMPRESSION_FAILED_INFLATED_TOKEN_COUNT,

  /** The compression failed due to an error counting tokens */
  COMPRESSION_FAILED_TOKEN_COUNT_ERROR,

  /** The compression was not necessary and no action was taken */
  NOOP,
}

export interface ChatCompressionInfo {
  originalTokenCount: number;
  newTokenCount: number;
  compressionStatus: CompressionStatus;
}

export type ServerKaiDexChatCompressedEvent = {
  type: KaiDexEventType.ChatCompressed;
  value: ChatCompressionInfo | null;
};

export type ServerKaiDexMaxSessionTurnsEvent = {
  type: KaiDexEventType.MaxSessionTurns;
};

export type ServerKaiDexFinishedEvent = {
  type: KaiDexEventType.Finished;
  value: KaiDexFinishedEventValue;
};

export type ServerKaiDexLoopDetectedEvent = {
  type: KaiDexEventType.LoopDetected;
};

export type ServerKaiDexCitationEvent = {
  type: KaiDexEventType.Citation;
  value: string;
};

// The original union type, now composed of the individual types
export type ServerKaiDexStreamEvent =
  | ServerKaiDexChatCompressedEvent
  | ServerKaiDexCitationEvent
  | ServerKaiDexContentEvent
  | ServerKaiDexErrorEvent
  | ServerKaiDexFinishedEvent
  | ServerKaiDexLoopDetectedEvent
  | ServerKaiDexMaxSessionTurnsEvent
  | ServerKaiDexThoughtEvent
  | ServerKaiDexToolCallConfirmationEvent
  | ServerKaiDexToolCallRequestEvent
  | ServerKaiDexToolCallResponseEvent
  | ServerKaiDexUserCancelledEvent
  | ServerKaiDexRetryEvent;

// A turn manages the agentic loop turn within the server context.
export class Turn {
  pendingToolCalls: ToolCallRequestInfo[] = [];
  private debugResponses: GenerateContentResponse[] = [];
  private pendingCitations = new Set<string>();
  finishReason: FinishReason | undefined = undefined;

  constructor(
    private readonly chat: KaiDexChat,
    private readonly prompt_id: string,
  ) {}
  // The run method yields simpler events suitable for server logic
  async *run(
    req: PartListUnion,
    signal: AbortSignal,
  ): AsyncGenerator<ServerKaiDexStreamEvent> {
    try {
      // Note: This assumes `sendMessageStream` yields events like
      // { type: StreamEventType.RETRY } or { type: StreamEventType.CHUNK, value: GenerateContentResponse }
      const responseStream = await this.chat.sendMessageStream(
        {
          message: req as any,
          config: {
            abortSignal: signal,
          },
        },
        this.prompt_id,
      );

      for await (const streamEvent of responseStream) {
        console.log(
          streamEvent.type,
        );
        if (signal?.aborted) {
          yield { type: KaiDexEventType.UserCancelled };
          return;
        }

        // Handle the new RETRY event
        if (streamEvent.type === "retry") {
          yield { type: KaiDexEventType.Retry };
          continue; // Skip to the next event in the stream
        }

        // Assuming other events are chunks with a `value` property
        const resp = streamEvent.value as GenerateContentResponse;
        console.log(
          resp?.functionCalls?.length || 0,
        );
        if (!resp) continue; // Skip if there's no response body

        this.debugResponses.push(resp);

        const thoughtPart = resp.candidates?.[0]?.content?.parts?.[0];
        if ((thoughtPart as any)?.thought) {
          // Thought always has a bold "subject" part enclosed in double asterisks
          // (e.g., **Subject**). The rest of the string is considered the description.
          const rawText = thoughtPart?.text ?? "";
          const subjectStringMatches = rawText.match(/\*\*(.*?)\*\*/s);
          const subject = subjectStringMatches
            ? subjectStringMatches[1].trim()
            : "";
          const description = rawText.replace(/\*\*(.*?)\*\*/s, "").trim();
          const thought: ThoughtSummary = {
            subject,
            description,
          };

          yield {
            type: KaiDexEventType.Thought,
            value: thought,
          };
          continue;
        }

        const text = getResponseText(resp as any);
        if (text) {
          yield { type: KaiDexEventType.Content, value: text };
        }

        // Handle function calls (requesting tool execution)
        // Extract from parts to get thoughtSignature (Gemini 3 Pro requirement)
        const parts = resp.candidates?.[0]?.content?.parts ?? [];
        const functionCallParts = parts.filter(
          (p: any) => p.functionCall !== undefined,
        );
        console.log(functionCallParts.length);
        for (const part of functionCallParts) {
          const fnCall = (part as any).functionCall as FunctionCall;
          const thoughtSignature = (part as any).thoughtSignature as
            | string
            | undefined;
          console.log(fnCall.name, "args:", fnCall.args);
          const event = this.handlePendingFunctionCall(fnCall, thoughtSignature);
          if (event) {
            yield event;
            console.log();
          }
        }

        for (const citation of getCitations(resp)) {
          this.pendingCitations.add(citation);
        }

        // Check if response was truncated or stopped for various reasons
        const finishReason = resp.candidates?.[0]?.finishReason as any;

        // This is the key change: Only yield 'Finished' if there is a finishReason.
        if (finishReason) {
          if (this.pendingCitations.size > 0) {
            yield {
              type: KaiDexEventType.Citation,
              value: `Citations:\n${[...this.pendingCitations].sort().join("\n")}`,
            };
            this.pendingCitations.clear();
          }

          this.finishReason = finishReason as FinishReason;
          yield {
            type: KaiDexEventType.Finished,
            value: {
              reason: finishReason as FinishReason,
              usageMetadata: resp.usageMetadata,
            },
          };
        }
      }
    } catch (e) {
      if (signal.aborted) {
        yield { type: KaiDexEventType.UserCancelled };
        // Regular cancellation error, fail gracefully.
        return;
      }

      const error = toFriendlyError(e);
      if (error instanceof UnauthorizedError) {
        throw error;
      }

      const reqContent = createKaidexUserContent(req);
      const contextForReport = [
        ...this.chat.getHistory(/*curated*/ true),
        reqContent,
      ];
      await reportError(
        error,
        "Error when talking to KaiDex API",
        contextForReport,
        "Turn.run-sendMessageStream",
      );
      const status =
        typeof error === "object" &&
        error !== null &&
        "status" in error &&
        typeof (error as { status: unknown }).status === "number"
          ? (error as { status: number }).status
          : undefined;
      const structuredError: StructuredError = {
        message: getErrorMessage(error),
        status,
      };
      await this.chat.maybeIncludeSchemaDepthContext(structuredError);
      yield { type: KaiDexEventType.Error, value: { error: structuredError } };
      return;
    }
  }

  private handlePendingFunctionCall(
    fnCall: FunctionCall,
    thoughtSignature?: string,
  ): ServerKaiDexStreamEvent | null {
    const callId =
      fnCall.id ??
      `${fnCall.name}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const name = fnCall.name || "undefined_tool_name";
    const args = (fnCall.args || {}) as Record<string, unknown>;

    const toolCallRequest: ToolCallRequestInfo = {
      callId,
      name,
      args,
      isClientInitiated: false,
      prompt_id: this.prompt_id,
      thoughtSignature,
    };

    this.pendingToolCalls.push(toolCallRequest);

    // Yield a request for the tool call, not the pending/confirming status
    return { type: KaiDexEventType.ToolCallRequest, value: toolCallRequest };
  }

  getDebugResponses(): GenerateContentResponse[] {
    return this.debugResponses;
  }

  clearPendingToolCalls(): void {
    this.pendingToolCalls.length = 0;
  }
}

function getCitations(resp: GenerateContentResponse): string[] {
  return (resp.candidates?.[0]?.citationMetadata?.citations ?? [])
    .filter((citation: any) => citation.uri !== undefined)
    .map((citation: any) => {
      if (citation.title) {
        return `(${citation.title}) ${citation.uri}`;
      }
      return citation.uri!;
    });
}
