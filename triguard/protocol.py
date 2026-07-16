PROTOCOL_VERSION = "2.3"


def validate_protocol_values(values, source: str):
    observed = {str(value) for value in values if str(value)}
    if observed != {PROTOCOL_VERSION}:
        raise ValueError(
            f"{source} contains protocol versions {sorted(observed)}; "
            f"expected only {PROTOCOL_VERSION}. Use a fresh output directory."
        )
