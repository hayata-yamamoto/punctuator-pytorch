from typing import NoReturn, IO

from setuptools import find_packages, setup


def main() -> NoReturn:

    f: IO
    with open('sample-requirements.txt', 'r') as f:
        req = f.read().splitlines()

    setup(
        name="punctuator",
        description="Data Science Project. Have Fun!",
        version="0.1.0",
        install_requires=req,
        packages=find_packages(exclude=["data"])
    )

if __name__ == '__main__':
    main()
