"""Retry helpers: exponential backoff with jitter, retrying only transient errors."""
from __future__ import annotations

from typing import Callable, Tuple, Type, TypeVar

from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from agentflow.errors import TransientError

T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    initial: float = 0.5,
    max_wait: float = 8.0,
    retry_on: Tuple[Type[BaseException], ...] = (TransientError,),
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> Tuple[T, int]:
    """Run ``fn`` with retries. Returns ``(result, attempts_used)``.

    The final exception is re-raised unchanged if every attempt fails.
    """

    def _before_sleep(state: RetryCallState) -> None:
        if on_retry and state.outcome is not None:
            on_retry(state.attempt_number, state.outcome.exception())

    retrying = Retrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=initial, max=max_wait) + wait_random(0, initial),  # backoff + jitter
        retry=retry_if_exception_type(retry_on),
        before_sleep=_before_sleep,
        reraise=True,
    )
    used = 0
    for attempt in retrying:
        with attempt:
            used = attempt.retry_state.attempt_number
            result = fn()
    return result, used  # type: ignore[possibly-undefined]
