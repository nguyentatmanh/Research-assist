class ResearchError(Exception):
    """Safe, user-facing error. Never includes raw HTTP responses or credentials."""


class DocumentError(ResearchError):
    pass


class ProviderError(ResearchError):
    pass


class BudgetExceeded(ProviderError):
    """A run limit was reached; verified work can still be rendered locally."""


class ActionRejected(ResearchError):
    """Safe feedback for an agent to repair an invalid action or unmet dependency."""


class NotFoundError(ResearchError):
    pass


class CapacityError(ResearchError):
    pass
