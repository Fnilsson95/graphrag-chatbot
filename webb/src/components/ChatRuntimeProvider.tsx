/**
 * Wires assistant-ui to the backend ``/prompt/stream`` SSE endpoint.
 * ``VITE_API_BASE_URL`` defaults to the local FastAPI server.
 */
import type { FC, ReactNode } from "react";
import { useMemo } from "react";

import {
  AssistantRuntimeProvider,
  useLocalRuntime,
} from "@assistant-ui/react";

import { TooltipProvider } from "@/components/ui/tooltip";
import { createPromptApiAdapter } from "@/lib/prompt-api-adapter";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

type Props = {
  children: ReactNode;
};

export const ChatRuntimeProvider: FC<Props> = ({ children }) => {
  const adapter = useMemo(() => createPromptApiAdapter(API_BASE), []);
  const runtime = useLocalRuntime(adapter);
  return (
    <TooltipProvider>
      <AssistantRuntimeProvider runtime={runtime}>
        {children}
      </AssistantRuntimeProvider>
    </TooltipProvider>
  );
};
