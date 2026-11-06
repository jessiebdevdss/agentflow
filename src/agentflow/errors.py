"""Error taxonomy. Transient errors are retried with backoff; permanent ones are surfaced
to the agent (so it can self-correct) or trigger a graceful fallback."""


class AgentFlowError(Exception):
    """Base class for AgentFlow errors."""


class TransientError(AgentFlowError):
    """Temporary failure (timeout, 429, 5xx, connection reset). Safe to retry."""


class PermanentError(AgentFlowError):
    """Failure that will not go away on retry (bad input, 4xx, policy violation)."""


class LLMOutputError(AgentFlowError):
    """The model produced output that could not be parsed or validated."""


class CircuitOpenError(PermanentError):
    """A tool failed too many times in a row and has been disabled for this run."""
