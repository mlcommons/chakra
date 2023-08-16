#!/usr/bin/env python3

import os

from setuptools import setup

def main():
    package_base = "chakra"

    # List the packages and their dir mapping:
    # "install_destination_package_path": "source_dir_path"
    package_dir_map = {
        f"{package_base}": ".",
        f"{package_base}.third_party.utils": "third_party/utils",
        f"{package_base}.et_def": "et_def",
        f"{package_base}.et_converter": "et_converter",
        f"{package_base}.et_generator": "utils/et_generator",
        f"{package_base}.et_jsonizer": "utils/et_jsonizer",
        f"{package_base}.et_visualizer": "et_visualizer",
        f"{package_base}.timeline_visualizer": "timeline_visualizer"
    }

    packages = list(package_dir_map)

    os.system("protoc et_def.proto --proto_path et_def --python_out et_def")

    setup(
        name="chakra",
        python_requires=">=3.8",
        author="MLCommons",
        author_email="chakra@mlcommons.org",
        url="https://github.com/mlcommons/chakra",
        packages=packages,
        package_dir=package_dir_map
    )

if __name__ == "__main__":
    main()
