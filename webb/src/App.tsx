/** Root layout: full-height chat thread inside the assistant-ui runtime. */
import type { FC } from "react";

import { Thread } from "@/components/assistant-ui/thread";
import { ChatRuntimeProvider } from "@/components/ChatRuntimeProvider";

const App: FC = () => {
  return (
    <ChatRuntimeProvider>
      <div className="flex h-dvh min-h-0 flex-col bg-background">
        <main className="flex min-h-0 flex-1 flex-col overflow-hidden">
          <Thread />
        </main>
      </div>
    </ChatRuntimeProvider>
  );
};

export default App;
