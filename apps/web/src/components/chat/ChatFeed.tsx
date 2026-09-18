import Markdown from "react-markdown";
import { useEffect, useRef } from "react";
import type { FeedMessage } from "../../hooks/useChatSession";
import type { RequestRecord } from "../../lib/telemetry/request-record";
import { MessageActions } from "./MessageActions";

export function ChatFeed({ messages, active, select, retry }: {
  messages: FeedMessage[]; active: boolean; select: (id: string) => void; retry: (r: RequestRecord) => void;
}) {
  const feed = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  useEffect(() => {
    if (feed.current && following.current) feed.current.scrollTop = feed.current.scrollHeight;
  }, [messages]);
  return <div className="feed recess" aria-label="Conversation" ref={feed} onScroll={event => {
    const element = event.currentTarget;
    following.current = element.scrollHeight - element.scrollTop - element.clientHeight < 60;
  }}>
    {!messages.length && <div className="empty-channel"><span className="channel-icon" aria-hidden="true">⌁</span>
      <p className="label">CH-01 / INTERACTIVE INFERENCE</p><h2>Channel ready.</h2>
      <p>Send a prompt. Follow the response.<br />Inspect what the browser actually observes.</p>
      <div className="signal-path">BROWSER <span>→</span> GATEWAY <span>→</span> WORKER</div>
    </div>}
    {messages.map(message => message.role === "user"
      ? <article className="message user plate chamfer" key={message.id}><div className="face">
        <header className="message-head"><span>[USER_INPUT]</span></header><p className="user-text">{message.content}</p>
      </div></article>
      : <article className={`message plate chamfer ${message.request.status}`} key={message.id} aria-label={`Response ${message.request.sequence}`}>
        <div className="face"><header className="message-head"><span>[MODEL_RESPONSE // {String(message.request.sequence).padStart(2, "0")}]</span>
          <MessageActions request={message.request} active={active} inspect={() => select(message.id)} retry={() => retry(message.request)} />
        </header>
        <div className="markdown"><Markdown>{message.request.content}</Markdown></div>
        <p className="response-status" role="status">{message.request.status === "submitting"
          ? <><i className="led checking pending" aria-hidden="true" /> AWAITING FIRST CONTENT</>
          : message.request.status === "cancelled" ? "ABORTED / PARTIAL RESPONSE RETAINED" : message.request.status.toUpperCase()}</p>
        {message.request.error && <div className="error-frame" role="alert">{message.request.error.code ? `HTTP ${message.request.error.code} / ` : ""}{message.request.error.message}
          <button disabled={active} onClick={() => retry(message.request)}>Retry</button></div>}
        {message.request.warnings.map(w => <p className="warning" key={w}>{w}</p>)}
        </div>
      </article>)}
  </div>;
}
