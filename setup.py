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
        f"{package_base}.eg_def": "eg_def",
        f"{package_base}.eg_converter": "eg_converter",
        f"{package_base}.eg_visualizer": "eg_visualizer",
        f"{package_base}.timeline_visualizer": "timeline_visualizer"
    }

    packages = list(package_dir_map)

    os.system("protoc eg_def.proto --proto_path eg_def --python_out eg_def")

    setup(
        name="chakra",
        python_requires=">=3.8",
        author="Taekyung Heo",
        author_email="taekyungheo@meta.com",
        url="https://github.com/chakra-eg/chakra",
        packages=packages,
        package_dir=package_dir_map
    )

if __name__ == "__main__":
    main()
