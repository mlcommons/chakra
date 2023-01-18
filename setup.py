#!/usr/bin/env python3

from setuptools import setup

def main():
    package_base = "chakra"

    # List the packages and their dir mapping:
    # "install_destination_package_path": "source_dir_path"
    package_dir_map = {
        f"{package_base}": ".",
        f"{package_base}.third_party.utils": "third_party/utils"
    }

    packages = list(package_dir_map)

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
