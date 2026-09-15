"""Streamlit interface for Stella.

A second front end over the same engine. Every piece of logic — turn
orchestration, extraction, scoring, state — is imported from the `stella`
package, exactly as `cli.py` imports it. Nothing about the conversation or the
confidence model is reimplemented here; this module only renders. If the two
front ends ever disagreed about a score, that would mean logic had leaked into
a view, which is precisely what this split is meant to prevent.

Every string here is UI chrome — headings, labels, button text, the confidence
gauge — never a conversational sentence. Model-authored text (chat bubbles,
the recommendation's reasoning and caveats) flows through untouched.

Run with:  streamlit run app.py
The CLI remains the primary entrypoint:  python run.py
"""

from __future__ import annotations

import html

import streamlit as st
from dotenv import load_dotenv

from stella import prompts
from stella.confidence import CONFIDENT_THRESHOLD, NARROWING_THRESHOLD, breakdown
from stella.conversation import Conversation
from stella.llm import DEFAULT_MODEL, LLMClient, LLMUnavailable
from stella.schemas import Recommendation

load_dotenv()

st.set_page_config(page_title="Stella — Size & Fit Advisor", page_icon="🪞", layout="centered")

# Known-good OpenRouter slugs. Free models cost nothing but are slower and
# noticeably weaker at the extraction step — see the README's model notes.
MODEL_CHOICES = [
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-haiku-4.5",
    "dots-studio/dots-3-note-preview:free",
    "nex-agi/nex-n2.5-pro:free",
]

# Muted, editorial palette — not the default red/amber/green, deliberately
# desaturated to sit on a warm cream background without looking like a
# dashboard alert.
GAUGE_COLORS = {
    "Guessing": "#B5654F",   # clay
    "Narrowing": "#C1932B",  # ochre
    "Confident": "#4B6B4E",  # sage
}

ASSISTANT_AVATAR = "🪡"


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', -apple-system, sans-serif;
        }

        .block-container {
            max-width: 760px;
            padding-top: 2.5rem;
        }

        h1, h2, h3, .stella-wordmark {
            font-family: 'Playfair Display', Georgia, serif !important;
        }

        .stella-hero {
            text-align: center;
            margin-bottom: 0.25rem;
        }
        .stella-wordmark {
            font-size: 3rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            color: #2A241F;
            margin: 0;
        }
        .stella-tagline {
            font-family: 'Inter', sans-serif;
            font-size: 0.95rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #9C4A3B;
            margin-top: 0.35rem;
        }
        .stella-rule {
            border: none;
            border-top: 1px solid #E4D9C9;
            margin: 1.75rem 0 1.5rem 0;
        }

        [data-testid="stChatMessage"] {
            background: #FFFFFF;
            border: 1px solid #ECE2D4;
            border-radius: 14px;
            padding: 0.25rem 0.5rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 1px 2px rgba(42, 36, 31, 0.04);
        }

        [data-testid="stChatInput"] {
            border-radius: 14px;
        }

        [data-testid="stSidebar"] {
            background: #F1E9DF;
            border-right: 1px solid #E4D9C9;
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            font-family: 'Playfair Display', Georgia, serif !important;
            color: #2A241F;
        }

        .stella-gauge-wrap {
            background: #FFFFFF;
            border: 1px solid #ECE2D4;
            border-radius: 14px;
            padding: 1rem 1.1rem 1.15rem 1.1rem;
            margin-bottom: 1rem;
        }
        .stella-gauge-label {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 1.05rem;
            font-weight: 600;
        }
        .stella-gauge-score {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            color: #6B6259;
            float: right;
        }
        .stella-gauge-track {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: #EFE7DA;
            overflow: hidden;
            margin-top: 0.55rem;
        }
        .stella-gauge-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.4s ease;
        }
        .stella-gauge-delta {
            font-size: 0.78rem;
            color: #6B6259;
            margin-top: 0.4rem;
        }

        .stella-slot-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            font-size: 0.75rem;
            margin-top: 0.4rem;
        }
        .stella-slot-table th {
            text-align: left;
            font-weight: 600;
            color: #6B6259;
            border-bottom: 1px solid #E4D9C9;
            padding: 0.3rem 0.15rem;
        }
        .stella-slot-table th:nth-child(1) { width: 40%; }
        .stella-slot-table th:nth-child(2) { width: 15%; text-align: right; }
        .stella-slot-table th:nth-child(3) { width: 27%; }
        .stella-slot-table th:nth-child(4) { width: 18%; text-align: right; }
        .stella-slot-table td {
            padding: 0.35rem 0.15rem;
            border-bottom: 1px solid #F1E9DF;
            color: #2A241F;
            word-break: break-word;
            vertical-align: top;
        }
        .stella-slot-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
        .stella-badge {
            display: inline-block;
            font-size: 0.62rem;
            padding: 0.02rem 0.3rem;
            border-radius: 999px;
            background: #F1E9DF;
            color: #9C4A3B;
            margin-left: 0.2rem;
        }

        .stella-note {
            font-size: 0.78rem;
            color: #9C4A3B;
            margin-top: 0.3rem;
        }

        .stella-rec-card {
            background: #FFFFFF;
            border: 1px solid #ECE2D4;
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            margin-top: 1rem;
            box-shadow: 0 2px 10px rgba(42, 36, 31, 0.06);
        }
        .stella-rec-title {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 1.3rem;
            font-weight: 700;
            color: #2A241F;
            margin-bottom: 0.9rem;
        }
        .stella-rec-row {
            display: flex;
            gap: 1rem;
            margin-bottom: 0.8rem;
        }
        .stella-rec-field {
            flex: 1;
        }
        .stella-rec-field-label {
            font-size: 0.72rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #9C4A3B;
            margin-bottom: 0.15rem;
        }
        .stella-rec-field-value {
            font-size: 1rem;
            color: #2A241F;
        }
        .stella-rec-tip {
            font-size: 0.92rem;
            color: #2A241F;
            padding-top: 0.6rem;
            border-top: 1px solid #F1E9DF;
        }
        .stella-caveats {
            margin-top: 0.9rem;
            padding-top: 0.7rem;
            border-top: 1px solid #F1E9DF;
        }
        .stella-caveats-label {
            font-size: 0.72rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #6B6259;
            margin-bottom: 0.3rem;
        }
        .stella-caveat {
            font-size: 0.85rem;
            color: #6B6259;
            padding-left: 1rem;
            position: relative;
            margin-bottom: 0.25rem;
        }
        .stella-caveat::before {
            content: "—";
            position: absolute;
            left: 0;
            color: #C1932B;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _compact(markup: str) -> str:
    """Collapse HTML markup to a single line before handing it to st.markdown.

    st.markdown parses its input as Markdown first, and Markdown treats any
    line indented four or more spaces as a code block. Interpolating an
    already-indented fragment (like generated table rows) into an indented
    template leaves exactly that, so the HTML renders as literal text instead
    of markup. Emitting one line sidesteps the whole class of bug.
    """
    return "".join(line.strip() for line in markup.splitlines())


def _gauge_color(label: str) -> str:
    return GAUGE_COLORS.get(label, "#9C4A3B")


def _render_hero() -> None:
    st.markdown(
        _compact(
            """
            <div class="stella-hero">
            <p class="stella-wordmark">Stella</p>
            <p class="stella-tagline">Size &amp; Fit Advisor</p>
            </div>
            <hr class="stella-rule" />
            """
        ),
        unsafe_allow_html=True,
    )


def _ensure_conversation(model: str) -> None:
    """Create the session and take the opening turn, exactly once."""
    if st.session_state.get("model") != model:
        st.session_state.model = model
        st.session_state.pop("conversation", None)

    if "conversation" in st.session_state:
        return

    try:
        client = LLMClient(model=model)
    except LLMUnavailable as exc:
        st.session_state.fatal = str(exc)
        return

    conversation = Conversation(client)
    st.session_state.conversation = conversation
    st.session_state.fatal = None
    st.session_state.error = None

    with st.spinner("Starting session…"):
        try:
            conversation.open()
        except LLMUnavailable as exc:
            st.session_state.error = str(exc)


def _render_recommendation(rec: Recommendation) -> None:
    caveats_html = ""
    if rec.caveats:
        items = "".join(f'<div class="stella-caveat">{html.escape(c)}</div>' for c in rec.caveats)
        caveats_html = (
            '<div class="stella-caveats">'
            '<div class="stella-caveats-label">Caveats</div>'
            f"{items}</div>"
        )

    st.markdown(
        _compact(
            '<div class="stella-rec-card">'
            '<div class="stella-rec-title">Your Fit</div>'
            '<div class="stella-rec-row">'
            '<div class="stella-rec-field">'
            '<div class="stella-rec-field-label">Size range</div>'
            f'<div class="stella-rec-field-value">{html.escape(rec.size_range)}</div>'
            "</div>"
            '<div class="stella-rec-field">'
            '<div class="stella-rec-field-label">Silhouette</div>'
            f'<div class="stella-rec-field-value">{html.escape(rec.silhouette)}</div>'
            "</div>"
            "</div>"
            '<div class="stella-rec-tip"><strong>Brand tip</strong> &mdash; '
            f"{html.escape(rec.brand_tip)}</div>"
            f"{caveats_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_sidebar(conversation: Conversation) -> None:
    state = conversation.state
    detail = breakdown(
        state.profile,
        state.contradictions,
        state.declined_slots,
        measurements_generic_only=state.measurements_generic_only,
    )
    score = detail["final"]
    label = detail["label"]
    color = _gauge_color(label)

    delta_html = ""
    if len(state.confidence_history) >= 2:
        change = state.confidence_history[-1] - state.confidence_history[-2]
        if change:
            arrow = "&#9650;" if change > 0 else "&#9660;"
            delta_html = f'<div class="stella-gauge-delta">{arrow} {change:+.1f} this turn</div>'

    with st.sidebar:
        st.markdown(
            _compact(
                '<div class="stella-gauge-wrap">'
                f'<span class="stella-gauge-label">{html.escape(label)}</span>'
                f'<span class="stella-gauge-score">{score:.1f} / 100</span>'
                '<div class="stella-gauge-track">'
                f'<div class="stella-gauge-fill" style="width:{score}%; background:{color};"></div>'
                "</div>"
                f"{delta_html}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        st.caption(
            f"Guessing < {NARROWING_THRESHOLD:.0f} · "
            f"Narrowing < {CONFIDENT_THRESHOLD:.0f} · "
            f"Confident ≥ {CONFIDENT_THRESHOLD:.0f}"
        )

        st.markdown("**Where the score comes from**")
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(row['slot'].replace('_', ' '))}"
            + ('<span class="stella-badge">amb</span>' if row["ambiguous"] else "")
            + "</td>"
            f'<td class="num">{row["weight"]:.2f}</td>'
            f"<td>{html.escape(row['specificity'])}</td>"
            f'<td class="num">{row["points"]:.1f}</td>'
            "</tr>"
            for row in detail["slots"]
        )
        st.markdown(
            _compact(
                '<table class="stella-slot-table">'
                "<tr><th>Slot</th><th>W</th><th>Signal</th><th>Pts</th></tr>"
                f"{rows}"
                "</table>"
            ),
            unsafe_allow_html=True,
        )

        notes = []
        if detail["contradiction_penalty"]:
            notes.append(f"−{detail['contradiction_penalty']:.1f} from contradictions")
        if detail["generic_size_cap_applied"]:
            notes.append("generic-size cap applied to measurements")
        if detail["declined_ceiling_applied"]:
            notes.append("declined-slot ceiling of 80 applied")
        for note in notes:
            st.markdown(f'<div class="stella-note">{html.escape(note)}</div>', unsafe_allow_html=True)

        if state.contradictions:
            st.markdown("**Unresolved contradictions**")
            for contradiction in state.contradictions:
                st.caption(contradiction.describe())

        with st.expander("Raw session state"):
            payload = state.to_dict()
            payload["prompt_versions"] = prompts.PROMPT_VERSIONS
            payload["model"] = st.session_state.model
            st.json(payload)

        if st.button("Reset session", use_container_width=True):
            st.session_state.pop("conversation", None)
            st.rerun()


_inject_style()
_render_hero()

selected_model = st.selectbox(
    "Model",
    MODEL_CHOICES,
    index=MODEL_CHOICES.index(DEFAULT_MODEL) if DEFAULT_MODEL in MODEL_CHOICES else 0,
    help="Free models cost nothing but are slower and weaker at extraction, "
    "which is the step the confidence score depends on.",
)

_ensure_conversation(selected_model)

if st.session_state.get("fatal"):
    st.error(st.session_state.fatal)
    st.stop()

conversation: Conversation = st.session_state.conversation

_render_sidebar(conversation)

for turn in conversation.state.transcript:
    avatar = ASSISTANT_AVATAR if turn.role == "assistant" else None
    with st.chat_message("assistant" if turn.role == "assistant" else "user", avatar=avatar):
        st.write(turn.content)

if conversation.state.recommendation is not None:
    _render_recommendation(Recommendation(**conversation.state.recommendation))

if st.session_state.get("error"):
    st.warning(f"{st.session_state.error}  Your answers so far are safe — try again.")

if conversation.finished:
    st.success("Session complete. Reset in the sidebar to start another.")
else:
    if reply := st.chat_input("Your answer…"):
        st.session_state.error = None
        with st.spinner("Thinking…"):
            try:
                conversation.handle(reply)
            except LLMUnavailable as exc:
                st.session_state.error = str(exc)
            except Exception as exc:  # noqa: BLE001 - the UI must never hard-crash
                st.session_state.error = f"Unexpected error ({type(exc).__name__}): {exc}"
        st.rerun()
