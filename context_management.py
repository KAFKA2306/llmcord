from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

Message = dict[str, Any]
TokenCounter = Callable[[list[Message]], Awaitable[int]]
Compactor = Callable[[list[Message], int], Awaitable[str]]

COMPACTION_SYSTEM_PROMPT = """You compact a conversation so it can continue without losing operationally important state.
Preserve facts; do not invent or reinterpret them. Keep concrete names, IDs, URLs, numbers, configuration values,
user goals and constraints, decisions and their reasons, completed work, unresolved blockers, failed approaches that
should not be repeated, next actions, and important facts extracted from attachments. Keep conflicts or uncertainty
explicit. Do not copy the system prompt into the summary.

Return only a concise structured summary with these headings when applicable:
- Goal
- Constraints
- Decisions and rationale
- Concrete facts
- Completed work
- Failed approaches / do not repeat
- Open items / blockers
- Next actions
""".strip()

SUMMARY_PREFIX = "[Compacted conversation state; older raw Discord history remains the source of record]\n"
CURRENT_INPUT_PREFIX = "[Current user input was automatically compacted to fit the model context]\n"


class ContextManagementError(RuntimeError):
    """Raised when context cannot be made safe without silent data loss."""


@dataclass(frozen=True)
class ContextPolicy:
    context_window_tokens: int
    max_output_tokens: int
    safety_margin_tokens: int
    compaction_trigger_tokens: int
    compaction_target_tokens: int
    recent_messages: int
    compaction_max_output_tokens: int

    def __post_init__(self) -> None:
        positive = {
            "context_window_tokens": self.context_window_tokens,
            "max_output_tokens": self.max_output_tokens,
            "compaction_trigger_tokens": self.compaction_trigger_tokens,
            "compaction_target_tokens": self.compaction_target_tokens,
            "recent_messages": self.recent_messages,
            "compaction_max_output_tokens": self.compaction_max_output_tokens,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ContextManagementError(f"{name} must be a positive integer")
        if not isinstance(self.safety_margin_tokens, int) or isinstance(self.safety_margin_tokens, bool) or self.safety_margin_tokens < 0:
            raise ContextManagementError("safety_margin_tokens must be a non-negative integer")
        if self.hard_input_limit <= 0:
            raise ContextManagementError("max_output_tokens + safety_margin_tokens must be smaller than context_window_tokens")
        if self.compaction_max_output_tokens + self.safety_margin_tokens >= self.context_window_tokens:
            raise ContextManagementError("compaction_max_output_tokens leaves no room for compaction input")
        if self.compaction_trigger_tokens > self.hard_input_limit:
            raise ContextManagementError("compaction_trigger_tokens must not exceed the hard input limit")
        if self.compaction_target_tokens >= self.compaction_trigger_tokens:
            raise ContextManagementError("compaction_target_tokens must be smaller than compaction_trigger_tokens")

    @property
    def hard_input_limit(self) -> int:
        return self.context_window_tokens - self.max_output_tokens - self.safety_margin_tokens

    @property
    def compaction_input_limit(self) -> int:
        return self.context_window_tokens - self.compaction_max_output_tokens - self.safety_margin_tokens


@dataclass(frozen=True)
class ContextResult:
    messages: list[Message]
    input_tokens_before: int
    input_tokens_after: int
    compacted: bool = False
    compacted_current_input: bool = False


def compaction_request_messages(history: list[Message]) -> list[Message]:
    return [
        {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": "Compact the conversation state now."},
    ]


def _is_system(message: Message) -> bool:
    return message.get("role") in {"system", "developer"}


async def _split_text_message(
    message: Message,
    input_limit: int,
    count_tokens: TokenCounter,
) -> list[Message]:
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ContextManagementError("a single non-text message is too large to compact safely")

    parts: list[Message] = []
    remaining = content
    while remaining:
        low, high, best = 1, len(remaining), 0
        while low <= high:
            middle = (low + high) // 2
            candidate = dict(message)
            candidate["content"] = remaining[:middle]
            tokens = await count_tokens(compaction_request_messages([candidate]))
            if tokens <= input_limit:
                best = middle
                low = middle + 1
            else:
                high = middle - 1

        if best == 0:
            raise ContextManagementError("token overhead leaves no room to compact a text fragment")

        fragment = dict(message)
        fragment["content"] = remaining[:best]
        parts.append(fragment)
        remaining = remaining[best:]

    return parts


async def _partition_for_compaction(
    history: list[Message],
    input_limit: int,
    count_tokens: TokenCounter,
) -> list[list[Message]]:
    chunks: list[list[Message]] = []
    current: list[Message] = []

    for message in history:
        candidate = [*current, message]
        if await count_tokens(compaction_request_messages(candidate)) <= input_limit:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = []

        if await count_tokens(compaction_request_messages([message])) <= input_limit:
            current = [message]
            continue

        fragments = await _split_text_message(message, input_limit, count_tokens)
        chunks.extend([[fragment] for fragment in fragments])

    if current:
        chunks.append(current)

    return chunks


async def _compact_history(
    history: list[Message],
    policy: ContextPolicy,
    count_tokens: TokenCounter,
    compact: Compactor,
) -> str:
    if not history:
        raise ContextManagementError("cannot compact an empty history")

    chunks = await _partition_for_compaction(history, policy.compaction_input_limit, count_tokens)
    summaries = [await compact(chunk, policy.compaction_max_output_tokens) for chunk in chunks]

    if any(not summary.strip() for summary in summaries):
        raise ContextManagementError("compaction returned an empty summary")

    while len(summaries) > 1:
        summary_messages = [
            {"role": "assistant", "content": f"[Partial compacted state]\n{summary}"}
            for summary in summaries
        ]
        chunks = await _partition_for_compaction(summary_messages, policy.compaction_input_limit, count_tokens)
        next_summaries = [await compact(chunk, policy.compaction_max_output_tokens) for chunk in chunks]
        if len(next_summaries) >= len(summaries) and len(summaries) > 1:
            raise ContextManagementError("compaction did not reduce the number of summary chunks")
        summaries = next_summaries

    return summaries[0].strip()


async def prepare_context(
    messages: list[Message],
    policy: ContextPolicy,
    count_tokens: TokenCounter,
    compact: Compactor,
) -> ContextResult:
    """Return a context that fits the runtime window without silent truncation.

    System/developer messages are never summarized. Recent conversation messages remain verbatim when possible.
    If the current input alone cannot fit, it is compacted rather than asking the user to resend a shorter message.
    """

    before = await count_tokens(messages)
    if before <= policy.compaction_trigger_tokens:
        return ContextResult(messages=list(messages), input_tokens_before=before, input_tokens_after=before)

    authority = [message for message in messages if _is_system(message)]
    conversation = [message for message in messages if not _is_system(message)]
    if not conversation:
        raise ContextManagementError("system/developer messages exceed the model context budget")

    # Prefer the configured recent window, but progressively compact older recent turns if required.
    # The target keeps active context focused; the hard limit remains the final safety boundary.
    best_safe: tuple[list[Message], int] | None = None
    for keep_count in range(min(policy.recent_messages, len(conversation)), 0, -1):
        older = conversation[:-keep_count]
        recent = conversation[-keep_count:]
        if not older:
            continue

        summary = await _compact_history(older, policy, count_tokens, compact)
        candidate = [
            *authority,
            {"role": "assistant", "content": SUMMARY_PREFIX + summary},
            *recent,
        ]
        after = await count_tokens(candidate)
        if after <= policy.hard_input_limit:
            best_safe = (candidate, after)
            if after <= policy.compaction_target_tokens:
                return ContextResult(
                    messages=candidate,
                    input_tokens_before=before,
                    input_tokens_after=after,
                    compacted=True,
                )

    if best_safe is not None:
        candidate, after = best_safe
        return ContextResult(
            messages=candidate,
            input_tokens_before=before,
            input_tokens_after=after,
            compacted=True,
        )

    # At this point the system/developer authority plus the current message itself is too large.
    # Compact the current input automatically instead of assigning context management to the user.
    current = conversation[-1]
    current_summary = await _compact_history([current], policy, count_tokens, compact)
    compacted_current = dict(current)
    compacted_current["content"] = CURRENT_INPUT_PREFIX + current_summary
    candidate = [*authority, compacted_current]
    after = await count_tokens(candidate)
    if after <= policy.hard_input_limit:
        return ContextResult(
            messages=candidate,
            input_tokens_before=before,
            input_tokens_after=after,
            compacted=True,
            compacted_current_input=True,
        )

    raise ContextManagementError("context remains larger than the runtime window after automatic compaction")
