from setuptools import find_packages, setup

setup(
    name="mi_proyecto_ml",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
