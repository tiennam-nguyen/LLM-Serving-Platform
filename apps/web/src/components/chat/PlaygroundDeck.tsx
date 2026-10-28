"use client";
import { useChatSession } from "../../hooks/useChatSession";
import { Plate } from "../shell/AppFrame";
import { TopBar } from "../shell/TopBar";
import { RequestInspector } from "../telemetry/RequestInspector";
import { ChatFeed } from "./ChatFeed";
import { Composer } from "./Composer";

export function PlaygroundDeck() {
  const model = process.env.NEXT_PUBLIC_DEFAULT_MODEL?.trim() || "";
  const session = useChatSession(model);
  return <>
    <TopBar model={model} health={session.health} />
    <div className="workspace-caption etch"><span>OPERATOR CONSOLE <b>/ PLAYGROUND</b></span><span>WEB → GATEWAY → INFERENCE</span></div>
    <main className="workspace">
      <Plate className="chat-deck"><header className="panel-header"><h2><i className="led checking" aria-hidden="true" /> CH-01: LOCAL_INTERACTIVE</h2>
        <button className="industrial-button" disabled={session.active || !session.messages.length} onClick={session.clear}>CLEAR CHAT</button></header>
        {!model && <div className="config-notice">MODEL NOT CONFIGURED — set <code>NEXT_PUBLIC_DEFAULT_MODEL</code> to the exact model served by your worker, then restart the web app.</div>}
        <ChatFeed messages={session.messages} active={session.active} select={session.select} retry={session.retry} />
        <Composer active={session.active} configured={!!model} onSend={session.send} onAbort={session.abort} />
      </Plate>
      <RequestInspector request={session.selected} requests={session.requests} select={session.select} />
    </main>
    <footer className="chassis-footer etch"><span>LLM / SERVING PLATFORM</span><span>CLIENT OBSERVATIONS · SESSION MEMORY ONLY</span></footer>
  </>;
}
