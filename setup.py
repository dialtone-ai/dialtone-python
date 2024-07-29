import toml
from setuptools import setup, find_packages

# This script builds the package

def convert_version_specifier(version):
    if version.startswith("^") or version.startswith("~"):
        return version[1:]
    return version


with open("pyproject.toml", "r") as f:
    pyproject = toml.load(f)

dependencies = pyproject["tool"]["poetry"]["dependencies"]
dependency_list = [
    f"{pkg}{convert_version_specifier(ver)}"
    for pkg, ver in dependencies.items()
    if pkg != "python"
]

setup(
    name="dialtone",
    version=pyproject["tool"]["poetry"]["version"],
    packages=find_packages(include=["dialtone.*"]),
    install_requires=dependency_list,
    author=pyproject["tool"]["poetry"]["authors"][0],
    description=pyproject["tool"]["poetry"]["description"],
)
