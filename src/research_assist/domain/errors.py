class ResearchError(Exception):
    """Safe, user-facing error. Never includes raw HTTP responses or credentials."""


class DocumentError(ResearchError):
    pass


class ProviderError(ResearchError):
    pass


class NotFoundError(ResearchError):
    pass


class CapacityError(ResearchError):
    pass
