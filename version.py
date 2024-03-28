"""Build utilities

"""
import subprocess
from typing import List
import sys
import os
import os.path
import argparse


def main(args: List[str] = None):
    """Build utilities

    args:
    --version-change: piece to change in semantic version number
    --build-file: the builder number filename which should cotain an single int
        that will be inremented by this utility
    """
    parser = argparse.ArgumentParser(
        prog="build",
        description="Build Utilities",
        epilog="Required",
    )
    parser.add_help = True
    parser.add_argument(
        "--version-change",
        type=str,
        choices=["major", "minor", "patch"],
        required=False,
        help="""
        --version-change == major
            Major version X (X.y.z | X > 0) MUST be incremented if any backward incompatible changes are introduced to the public API.
            It MAY also include minor and patch level changes.
            Patch and minor versions MUST be reset to 0 when major version is incremented.

        --version-change == minor
            Minor version Y (x.Y.z | x > 0) MUST be incremented if new, backward compatible functionality is introduced to the public API.
            It MUST be incremented if any public API functionality is marked as deprecated.
            It MAY be incremented if substantial new functionality or improvements are introduced within the private code.
            It MAY include patch level changes.
            Patch version MUST be reset to 0 when minor version is incremented.

        --version-change == patch
            Patch version Z (x.y.Z | x > 0) MUST be incremented if only backward compatible bug fixes are introduced.
            A bug fix is defined as an internal change that fixes incorrect behavior.
        """,
    )
    parser.add_argument(
        "--package",
        type=str,
        required=False,
        default=None,
        help="""Name of the package that will form part of the pre-release tag.
        If not supplied then no pre-release tag will be appended to the artifact.
        If supplied then a build number will also be added to the end of the
        pre-release tag""",
    )
    parser.add_argument(
        "--build",
        type=str,
        required=False,
        default=None,
        help="""Name of the file that contains the existing build number.
        If it does not exist a new one will be created with the build number starting at 1.""",
    )
    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    if os.path.exists("VERSION") is True:
        with open("VERSION", "rt", encoding="ascii") as fp:
            version_number = fp.read()
    else:
        version_number = "0.0.1"
    try:
        major, minor, patch = version_number.split(".")
    except Exception:
        major, minor, patch = "0.0.1".split(".")
    major = int(major)
    minor = int(minor)
    patch = int(patch)
    original_version_number = f"{major}.{minor}.{patch}"
    print(f'args.version_change: {args.version_change}')
    if args.version_change == "major":
        major += 1
        minor = 0
        patch = 0
    elif args.version_change == "minor":
        minor += 1
        patch = 0
    elif args.version_change == "patch":
        patch += 1
    new_version_number = f"{major}.{minor}.{patch}"
    print(
        f"current version:{original_version_number} change:{args.version_change} next version:{new_version_number}"
    )
    if original_version_number != new_version_number:
        with open("VERSION", "w", encoding="ascii") as fp:
            fp.write(new_version_number)

if __name__ == "__main__":
    main(sys.argv[1:])
