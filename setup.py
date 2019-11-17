from typing import NoReturn, IO

from setuptools import find_packages, setup


def main() -> NoReturn:

    setup(name="punctuator",
          description="Data Science Project. Have Fun!",
          version="0.1.0",
          install_requires=[],
          packages=find_packages(exclude=["data"]))


if __name__ == '__main__':
    main()
