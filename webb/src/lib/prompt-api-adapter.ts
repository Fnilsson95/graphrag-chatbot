/**
 * Bridges assistant-ui's ``ChatModelAdapter`` to the FastAPI ``/prompt/stream``
 * SSE endpoint. Manages per-thread conversation IDs and incremental rendering.
 */
import type {
  ChatModelAdapter,
  ChatModelRunResult,
  MessageTiming,
  ThreadMessage,
} from "@assistant-ui/react";

function lastUserText(messages: readonly ThreadMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]!;
    if (m.role !== "user") continue;
    const text = m.content
      .filter((p): p is { type: "text"; text: string } => p.type === "text")
      .map((p) => p.text)
      .join("");
    if (text) return text;
  }
  return "";
}

function formatBody(message: string, options?: string[] | null): string {
  if (options?.length) {
    const bullets = options.map((opt) => `- ${opt}`).join("\n");
    return `${message}\n\n${bullets}`;
  }
  return message;
}

function createConversationID(): string {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }
  return `conversation-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

/** True when this send is the first user message in the current thread. */
function isFirstUserMessageInThread(
  messages: readonly ThreadMessage[],
): boolean {
  let userCount = 0;
  for (const m of messages) {
    if (m.role === "user") userCount += 1;
  }
  return userCount === 1;
}

/** Client-side round trip: send → full response (ms). */
function clientRoundTripTiming(startedAt: number): MessageTiming {
  return {
    streamStartTime: startedAt,
    totalStreamTime: performance.now() - startedAt,
    totalChunks: 1,
    toolCallCount: 0,
  };
}

/** Parse one SSE frame from a buffer, return [event, dataJSON] or null. */
type SseFrame = { event: string; data: unknown };
function parseFrame(raw: string): SseFrame | null {
  let event = "message";
  let dataLine = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLine += line.slice(5).trim();
  }
  if (!dataLine) return null;
  try {
    return { event, data: JSON.parse(dataLine) };
  } catch {
    return null;
  }
}

/** Consume an SSE response into discrete frames. */
async function* readSseFrames(
  res: Response,
): AsyncGenerator<SseFrame, void, unknown> {
  if (!res.body) return;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const frame = parseFrame(chunk);
      if (frame) yield frame;
    }
  }
}

/** Stream `POST /prompt/stream` and yield cumulative assistant content. */
export function createPromptApiAdapter(
  apiBaseUrl: string,
  conversationID = createConversationID(),
): ChatModelAdapter {
  const base = apiBaseUrl.replace(/\/$/, "");
  let activeConversationID = conversationID;
  return {
    async *run({ messages, abortSignal }) {
      const message = lastUserText(messages);
      if (isFirstUserMessageInThread(messages)) {
        activeConversationID = createConversationID();
      }
      const startedAt = performance.now();

      const res = await fetch(`${base}/prompt/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          message,
          conversationID: activeConversationID,
        }),
        signal: abortSignal,
      });

      if (!res.ok) {
        const errBody = await res.text();
        const result: ChatModelRunResult = {
          content: [
            {
              type: "text",
              text: `Request failed (${res.status}): ${errBody || res.statusText
                }`,
            },
          ],
          status: { type: "incomplete", reason: "error" },
          metadata: { timing: clientRoundTripTiming(startedAt) },
        };
        yield result;
        return;
      }

      let answer = "";
      let clarification: { message: string; options: string[] } | null = null;
      let errored: { status: number; detail: string } | null = null;

      for await (const frame of readSseFrames(res)) {
        const data = frame.data as Record<string, unknown>;
        switch (frame.event) {
          case "meta":
            if (typeof data.conversationID === "string") {
              activeConversationID = data.conversationID;
            }
            break;
          case "chunk":
            if (typeof data.text === "string") answer += data.text;
            yield {
              content: [{ type: "text", text: answer }],
              status: { type: "running" },
            };
            break;
          case "clarification":
            clarification = {
              message: String(data.message ?? ""),
              options: (data.options as string[]) ?? [],
            };
            break;
          case "error":
            errored = {
              status: Number(data.status ?? 500),
              detail: String(data.detail ?? "Unknown error"),
            };
            break;
          case "done":
            break;
        }
      }

      if (errored) {
        yield {
          content: [
            {
              type: "text",
              text: `Request failed (${errored.status}): ${errored.detail}`,
            },
          ],
          status: { type: "incomplete", reason: "error" },
          metadata: { timing: clientRoundTripTiming(startedAt) },
        };
        return;
      }

      const body = clarification
        ? formatBody(clarification.message, clarification.options)
        : answer;

      yield {
        content: [{ type: "text", text: body }],
        status: { type: "complete", reason: "stop" },
        metadata: { timing: clientRoundTripTiming(startedAt) },
      };
    },
  };
}
