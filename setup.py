from distutils.command.build import build

from setuptools import setup


class build_grpc(build):
    sub_commands = [("build_grpc", None)] + build.sub_commands


setup(cmdclass={"build": build_grpc})
