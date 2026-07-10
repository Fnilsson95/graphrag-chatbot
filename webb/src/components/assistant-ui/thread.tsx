/**
 * Chat thread UI built on assistant-ui primitives.
 * Renders the welcome screen, message list, composer, and round-trip timing.
 */
// @ts-nocheck — assistant-ui `render` slots vs base-ui component typings
import {
  ComposerAddAttachment,
  ComposerAttachments,
  UserMessageAttachments,
} from "@/components/assistant-ui/attachment";
import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { ToolFallback } from "@/components/assistant-ui/tool-fallback";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { Button } from "@/components/ui/button";
import {
  AuiIf,
  ComposerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  SuggestionPrimitive,
  ThreadPrimitive,
  useAuiState,
  useMessageTiming,
} from "@assistant-ui/react";
import { ArrowUpIcon, SquareIcon } from "lucide-react";
import type { FC } from "react";

export const Thread: FC = () => {
  return (
    <ThreadPrimitive.Root
      className="aui-root aui-thread-root @container flex h-full min-h-0 flex-1 flex-col bg-background"
      style={{
        ["--thread-max-width" as string]: "44rem",
      }}
    >
      <ThreadPrimitive.Viewport
        turnAnchor="bottom"
        className="aui-thread-viewport relative flex min-h-0 flex-1 flex-col overflow-x-hidden overflow-y-auto scroll-smooth px-4 sm:px-6"
      >
        <AuiIf condition={(s) => s.thread.isEmpty}>
          <ThreadWelcome />
        </AuiIf>

        <ThreadPrimitive.Messages>
          {() => <ThreadMessage />}
        </ThreadPrimitive.Messages>

        <AuiIf condition={(s) => s.thread.isRunning}>
          <AssistantTypingRow />
        </AuiIf>

        <ThreadPrimitive.ViewportFooter className="aui-thread-viewport-footer sticky bottom-0 z-10 mx-auto mt-auto flex w-full max-w-(--thread-max-width) shrink-0 flex-col bg-background px-2 pt-2 pb-[calc(1rem+env(safe-area-inset-bottom,0px))]">
          <Composer />
          <p className="mt-2 text-center text-muted-foreground text-xs">
            GraphRAG can make mistakes. Verify important information.
          </p>
        </ThreadPrimitive.ViewportFooter>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
};

const ThreadMessage: FC = () => {
  const role = useAuiState((s) => s.message.role);
  const isEditing = useAuiState((s) => s.message.composer.isEditing);
  if (isEditing) return <EditComposer />;
  if (role === "user") return <UserMessage />;
  return <AssistantMessage />;
};

const AssistantTypingRow: FC = () => {
  return (
    <div
      className="mx-auto w-full max-w-(--thread-max-width) shrink-0 px-1 py-4"
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <span className="sr-only">Assistant is replying</span>
      <div className="flex items-center gap-2 text-muted-foreground text-sm">
        <span className="flex items-center gap-1">
          <span className="chat-typing-dot size-1.5 rounded-full bg-foreground/60" />
          <span className="chat-typing-dot size-1.5 rounded-full bg-foreground/60" />
          <span className="chat-typing-dot size-1.5 rounded-full bg-foreground/60" />
        </span>
        <span>Thinking…</span>
      </div>
    </div>
  );
};

const ThreadWelcome: FC = () => {
  return (
    <div className="aui-thread-welcome-root mx-auto flex w-full max-w-(--thread-max-width) shrink-0 flex-col items-center px-1 pt-20 pb-4">
      <h1 className="text-center font-semibold text-3xl text-foreground tracking-tight">
        How can I help you today?
      </h1>
      <p className="mt-3 text-center text-base text-muted-foreground">
        Ask a question about the indexed documents to get started.
      </p>
      <ThreadSuggestions />
    </div>
  );
};

const ThreadSuggestions: FC = () => {
  return (
    <div className="aui-thread-welcome-suggestions mt-8 grid w-full @md:grid-cols-2 gap-2">
      <ThreadPrimitive.Suggestions>
        {() => <ThreadSuggestionItem />}
      </ThreadPrimitive.Suggestions>
    </div>
  );
};

const ThreadSuggestionItem: FC = () => {
  return (
    <div className="fade-in slide-in-from-bottom-2 @md:nth-[n+3]:block nth-[n+3]:hidden animate-in fill-mode-both duration-200">
      <SuggestionPrimitive.Trigger
        send
        render={
          <Button
            variant="outline"
            className="h-auto w-full cursor-pointer items-start justify-start gap-0.5 whitespace-normal rounded-xl border-border bg-card px-4 py-3 text-left text-sm @md:flex-col flex-wrap shadow-none transition-colors hover:bg-accent"
          />
        }
      >
        <SuggestionPrimitive.Title className="font-medium text-foreground" />
        <SuggestionPrimitive.Description className="text-muted-foreground text-xs empty:hidden" />
      </SuggestionPrimitive.Trigger>
    </div>
  );
};

const Composer: FC = () => {
  return (
    <ComposerPrimitive.Root className="aui-composer-root relative flex w-full flex-col">
      <ComposerPrimitive.AttachmentDropzone
        render={
          <div className="flex w-full flex-col gap-2 rounded-3xl border border-border bg-card px-4 py-3 shadow-sm transition-colors focus-within:border-foreground/20 focus-within:shadow-md data-[dragging=true]:border-foreground/40 data-[dragging=true]:border-dashed data-[dragging=true]:bg-accent" />
        }
      >
        <ComposerAttachments />
        <ComposerPrimitive.Input
          placeholder="Message GraphRAG…"
          className="aui-composer-input max-h-40 min-h-6 w-full resize-none bg-transparent text-[15px] text-foreground leading-6 outline-none placeholder:text-muted-foreground"
          rows={1}
          autoFocus
          aria-label="Message input"
        />
        <ComposerAction />
      </ComposerPrimitive.AttachmentDropzone>
    </ComposerPrimitive.Root>
  );
};

const ComposerAction: FC = () => {
  return (
    <div className="relative flex items-center justify-between">
      <ComposerAddAttachment />
      <AuiIf condition={(s) => !s.thread.isRunning}>
        <ComposerPrimitive.Send
          render={
            <TooltipIconButton
              tooltip="Send message"
              side="bottom"
              type="button"
              variant="default"
              size="icon"
              className="size-9 cursor-pointer rounded-full bg-foreground text-background hover:bg-foreground/85"
              aria-label="Send message"
            />
          }
        >
          <ArrowUpIcon className="size-4" />
        </ComposerPrimitive.Send>
      </AuiIf>
      <AuiIf condition={(s) => s.thread.isRunning}>
        <ComposerPrimitive.Cancel
          render={
            <Button
              type="button"
              variant="default"
              size="icon"
              className="size-9 cursor-pointer rounded-full"
              aria-label="Stop generating"
            />
          }
        >
          <SquareIcon className="size-3 fill-current" />
        </ComposerPrimitive.Cancel>
      </AuiIf>
    </div>
  );
};

const MessageError: FC = () => {
  return (
    <MessagePrimitive.Error>
      <ErrorPrimitive.Root className="mt-2 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-destructive text-sm">
        <ErrorPrimitive.Message className="line-clamp-2" />
      </ErrorPrimitive.Root>
    </MessagePrimitive.Error>
  );
};

const AssistantRoundTripTiming: FC = () => {
  const timing = useMessageTiming();
  const ms = timing?.totalStreamTime;
  if (ms == null) return null;
  const seconds = ms / 1000;
  const label =
    seconds >= 10 ? `${seconds.toFixed(0)} s` : `${seconds.toFixed(2)} s`;
  return (
    <span
      className="mt-1.5 block text-muted-foreground text-xs tabular-nums"
      title="Time from sending your message until the full reply was received"
    >
      {label}
    </span>
  );
};

const AssistantMessage: FC = () => {
  return (
    <MessagePrimitive.Root
      className="fade-in slide-in-from-bottom-1 relative mx-auto w-full max-w-(--thread-max-width) animate-in py-4 duration-150"
      data-role="assistant"
    >
      <div className="w-full text-[15px] text-foreground leading-relaxed">
        <MessagePrimitive.Parts>
          {({ part }) => {
            if (part.type === "text") return <MarkdownText />;
            if (part.type === "tool-call")
              return part.toolUI ?? <ToolFallback {...part} />;
            return null;
          }}
        </MessagePrimitive.Parts>
        <MessageError />
        <AssistantRoundTripTiming />
      </div>
    </MessagePrimitive.Root>
  );
};

const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root
      className="fade-in slide-in-from-bottom-1 mx-auto flex w-full max-w-(--thread-max-width) animate-in flex-col items-end gap-1.5 py-3 duration-150"
      data-role="user"
    >
      <UserMessageAttachments />
      <div className="max-w-[80%]">
        <div className="wrap-break-word rounded-3xl bg-muted px-4 py-2.5 text-[15px] text-foreground leading-relaxed empty:hidden">
          <MessagePrimitive.Parts />
        </div>
      </div>
    </MessagePrimitive.Root>
  );
};

const EditComposer: FC = () => {
  return (
    <MessagePrimitive.Root className="mx-auto flex w-full max-w-(--thread-max-width) flex-col items-end py-3">
      <ComposerPrimitive.Root className="flex w-full max-w-[80%] flex-col rounded-2xl border border-border bg-card">
        <ComposerPrimitive.Input
          className="min-h-14 w-full resize-none bg-transparent p-4 text-[15px] text-foreground outline-none"
          autoFocus
        />
        <div className="mx-3 mb-3 flex items-center gap-2 self-end">
          <ComposerPrimitive.Cancel render={<Button variant="ghost" size="sm" />}>
            Cancel
          </ComposerPrimitive.Cancel>
          <ComposerPrimitive.Send render={<Button size="sm" />}>
            Update
          </ComposerPrimitive.Send>
        </div>
      </ComposerPrimitive.Root>
    </MessagePrimitive.Root>
  );
};
