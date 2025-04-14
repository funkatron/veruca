from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="veruca",
    version="0.2.0",
    author="Chris Jones",
    author_email="chris@funkatron.com",
    description="A collection of tools for working with local LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/funkatron/veruca",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.12",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "veruca=veruca.cli:main",
        ],
    },
)