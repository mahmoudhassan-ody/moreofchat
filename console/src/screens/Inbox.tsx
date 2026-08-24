import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  claimConversation,
  listConversations,
  readThread,
  returnToBot,
  sendReply,
  type Conversation,
  type ThreadMessage,
} from "../api/inbox";
import { SourcePane } from "../components/SourcePane";

/**
 * Three panes: conversations, the thread, and where the figures came from.
 *
 * The thread is one contact's whole history across every channel they used —
 * a customer who asks on WhatsApp and follows up on Instagram is one person,
 * and an agent seeing only the channel the handoff fired on asks them
 * something they already answered.
 *
 * Selecting a bot message shows its sources. Selecting rather than always
 * showing the last one: the question an agent has is about a specific figure
 * in a specific reply, and a pane that tracks the newest message answers a
 * different question every time a new one arrives.
 */
export function Inbox() {
  const { t } = useTranslation();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selected, setSelected] = useState<Conversation | null>(null);
  const [thread, setThread] = useState<ThreadMessage[]>([]);
  const [shown, setShown] = useState<number | null>(null);
  const [draft, setDraft] = useState("");

  useEffect(() => {
    void listConversations().then(setConversations);
  }, []);

  async function open(conversation: Conversation) {
    setSelected(conversation);
    setShown(null);
    setThread(await readThread(conversation.id));
  }

  async function onClaim() {
    if (!selected) return;
    await claimConversation(selected.id);
    setConversations(await listConversations());
  }

  async function onSend() {
    if (!selected || !draft.trim()) return;
    await sendReply(selected.id, draft);
    setDraft("");
    setThread(await readThread(selected.id));
  }

  async function onReturn() {
    if (!selected) return;
    await returnToBot(selected.id);
    setConversations(await listConversations());
  }

  return (
    <div className="panes">
      <div className="pane list">
        <div className="pane-head">
          <span className="pane-title">{t("inbox.conversations")}</span>
        </div>
        {conversations.map((conversation) => (
          <div
            className={selected?.id === conversation.id ? "conv on" : "conv"}
            key={conversation.id}
            onClick={() => void open(conversation)}
          >
            <span className="ch">{conversation.channel.slice(0, 2).toUpperCase()}</span>
            <div className="conv-body">
              <div className="conv-name">{conversation.sender_ref}</div>
              <div className="conv-last">{conversation.reason}</div>
              {/* The routed team, as text rather than a second pill: one
                  accent on the screen, and it belongs to "needs a human". */}
              {conversation.team && (
                <div className="conv-team">
                  {t("inbox.routedTo")} {conversation.team}
                </div>
              )}
            </div>
            {/* The only coloured pill on the screen — something needs a human. */}
            <span className={conversation.claimed_by ? "pill bot" : "pill needs"}>
              {conversation.claimed_by ? t("inbox.claimed") : t("inbox.needsHuman")}
            </span>
          </div>
        ))}
      </div>

      <div className="pane thread">
        <div className="thread-head">
          <div className="who">
            {selected ? selected.sender_ref : t("inbox.pickOne")}
            {selected && <small>{selected.channel}</small>}
          </div>
          {selected && (
            <>
              <button className="act" type="button" onClick={() => void onClaim()}>
                {t("inbox.takeOver")}
              </button>
              <button className="act ghost" type="button" onClick={() => void onReturn()}>
                {t("inbox.returnToBot")}
              </button>
            </>
          )}
        </div>

        <div className="msgs">
          {thread.map((message, index) => (
            <div
              className={message.author === "customer" ? "row in" : "row out"}
              key={index}
              onClick={() => setShown(message.provenance ? index : null)}
            >
              <div className="bubble">{message.body}</div>
              {message.provenance && (
                <button className="why" type="button">
                  {t("inbox.why")}
                </button>
              )}
            </div>
          ))}
        </div>

        {selected && (
          <div className="composer">
            <textarea
              className="field"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder={t("inbox.reply")}
            />
            <button className="act" type="button" onClick={() => void onSend()}>
              {t("inbox.send")}
            </button>
          </div>
        )}
      </div>

      <SourcePane provenance={shown === null ? null : thread[shown].provenance} />
    </div>
  );
}
