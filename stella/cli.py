"""The REPL.

The only strings authored here are UI chrome: the progress bar, slash-command
help, error banners, and the labels on the final recommendation card. Every
conversational sentence comes from the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import prompts
from .confidence import breakdown, render_bar
from .conversation import Conversation, TurnResult
from .llm import LLMClient, LLMUnavailable
from .schemas import Recommendation
from .state import SessionState

BAR_WIDTH = 30

HELP_TEXT = """\
  /state       print the full session object as JSON
  /transcript  print the conversation so far
  /score       show the confidence arithmetic, slot by slot
  /save FILE   write the session to a JSON file
  /reset       start a new session
  /help        show this list
  /quit        exit
"""


def _banner(text: str) -> str:
    return f"\n  [!] {text}\n"


def _render_progress(result: TurnResult) -> str:
    delta = f"  ({result.delta:+.1f})" if result.delta else ""
    return (
        f"\n  {render_bar(result.score, BAR_WIDTH)}  "
        f"{result.score:.1f}%  {result.label}{delta}\n"
    )


def _render_recommendation(rec: Recommendation) -> str:
    lines = [
        "",
        "  " + "-" * 56,
        f"  SIZE RANGE   {rec.size_range}",
        f"  SILHOUETTE   {rec.silhouette}",
        f"  BRAND TIP    {rec.brand_tip}",
        "  " + "-" * 56,
    ]
    if rec.caveats:
        lines.append("  CAVEATS")
        lines.extend(f"    - {c}" for c in rec.caveats)
        lines.append("  " + "-" * 56)
    return "\n".join(lines)


def _render_score(state: SessionState) -> str:
    detail = breakdown(
        state.profile,
        state.contradictions,
        state.declined_slots,
        measurements_generic_only=state.measurements_generic_only,
    )
    rows = [f"  {'slot':<16}{'weight':>8}{'signal':>12}{'points':>9}"]
    for slot in detail["slots"]:
        flag = " (ambiguous)" if slot["ambiguous"] else ""
        rows.append(
            f"  {slot['slot']:<16}{slot['weight']:>8.2f}"
            f"{slot['specificity'] + flag:>12}{slot['points']:>9.1f}"
        )
    rows.append(f"  {'base':<16}{'':>8}{'':>12}{detail['base']:>9.1f}")
    if detail["contradiction_penalty"]:
        rows.append(
            f"  {'contradictions':<16}{'':>8}{'':>12}{-detail['contradiction_penalty']:>9.1f}"
        )
    if detail["generic_size_cap_applied"]:
        rows.append("  (generic-size cap applied to measurements)")
    if detail["declined_ceiling_applied"]:
        rows.append("  (declined-slot ceiling of 80 applied)")
    rows.append(f"  {'FINAL':<16}{'':>8}{'':>12}{detail['final']:>9.1f}  {detail['label']}")
    return "\n" + "\n".join(rows) + "\n"


class StellaCLI:
    def __init__(self, state: SessionState | None = None) -> None:
        self.client = LLMClient()
        self.conversation = Conversation(self.client, state)

    # ---------------------------------------------------------------- display

    def _show(self, result: TurnResult) -> None:
        print(f"\n  Stella: {result.text}")
        if result.recommendation is not None:
            print(_render_recommendation(result.recommendation))
        print(_render_progress(result), end="")

    # --------------------------------------------------------------- commands

    def _handle_command(self, line: str) -> bool:
        """Return False to exit the REPL."""
        parts = line.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""
        state = self.conversation.state

        if command in ("/quit", "/exit"):
            return False

        if command == "/help":
            print("\n" + HELP_TEXT)

        elif command == "/state":
            payload = state.to_dict()
            payload["prompt_versions"] = prompts.PROMPT_VERSIONS
            payload["model"] = self.client.model
            print("\n" + json.dumps(payload, indent=2) + "\n")

        elif command == "/transcript":
            if not state.transcript:
                print(_banner("Nothing said yet."))
            else:
                print()
                for turn in state.transcript:
                    print(f"  [{turn.timestamp}] {turn.role}: {turn.content}")
                print()

        elif command == "/score":
            print(_render_score(state))

        elif command == "/save":
            target = Path(argument or f"session_{state.session_id}.json")
            try:
                target.write_text(state.to_json(), encoding="utf-8")
                print(_banner(f"Saved to {target}"))
            except OSError as exc:
                print(_banner(f"Could not write {target}: {exc}"))

        elif command == "/reset":
            self.conversation = Conversation(self.client)
            print(_banner("New session started."))
            self._open()

        else:
            print(_banner(f"Unknown command {command}. Try /help."))

        return True

    # ------------------------------------------------------------------- loop

    def _open(self) -> None:
        try:
            self._show(self.conversation.open())
        except LLMUnavailable as exc:
            print(_banner(str(exc)))

    def run(self) -> int:
        print("\n  Stella — size & fit advisor.  /help for commands.\n")

        if not self.conversation.state.transcript:
            self._open()
        else:
            print(_banner("Resumed session. /transcript to see it."))

        while True:
            try:
                line = input("  You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            if not line:
                continue

            if line.startswith("/"):
                if not self._handle_command(line):
                    return 0
                continue

            if self.conversation.finished:
                print(_banner("This session is complete. /reset to start another, /quit to exit."))
                continue

            try:
                self._show(self.conversation.handle(line))
            except LLMUnavailable as exc:
                print(_banner(f"{exc} Your answers so far are safe — try again."))
            except KeyboardInterrupt:
                print(_banner("Interrupted."))
            except Exception as exc:  # noqa: BLE001 - the REPL must never die
                print(_banner(f"Unexpected error ({type(exc).__name__}): {exc}"))


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(prog="stella", description="CLI size & fit advisor.")
    parser.add_argument("--resume", metavar="FILE", help="resume a session saved with /save")
    args = parser.parse_args(argv)

    state = None
    if args.resume:
        try:
            state = SessionState.from_json(Path(args.resume).read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError) as exc:
            print(_banner(f"Could not resume {args.resume}: {exc}"))
            return 1

    try:
        cli = StellaCLI(state)
    except LLMUnavailable as exc:
        print(_banner(str(exc)))
        return 1

    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
