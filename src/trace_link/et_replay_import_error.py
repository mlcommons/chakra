"""Utility for handling et_replay import errors."""


def get_et_replay_install_error_msg() -> str:
    """
    Get the error message for missing et_replay installation.

    Returns
        str: Error message with installation instructions.
    """
    return (
        "Failed to import et_replay. et_replay is required but not packaged because it is not available as a PyPi package.\n\n"  # noqa: E501.
        "Please install it using:\n"
        '  pip install "git+https://github.com/facebookresearch/param.git@7b19f586dd8b267333114992833a0d7e0d601630#subdirectory=et_replay"\n\n'  # noqa: E501.
    )
