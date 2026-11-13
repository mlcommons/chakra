class HostTraceError(ValueError):
    """A host trace that cannot be read, or that lacks something a tool needs. The message names the file and node."""
