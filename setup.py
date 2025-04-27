from setuptools import setup, find_packages

setup(
    name="mnist_loader",
    version="0.1.0",
    package_dir={"": "src"},  # Indica que los paquetes están en src/
    packages=find_packages(where="src"),  # Busca paquetes en src/
    install_requires=[
         "numpy",          # Solo numpy (sin versión específica)
        "matplotlib",     # Solo matplotlib (sin versión específica)
        # Eliminé typing-extensions ya que no es esencial
    ],
    python_requires=">=3.12.3",
    author="Kent",
    author_email="corneigh@gmail.com",
    description="Paquete para carga y procesamiento de datos MNIST",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/embolao/mnist_loader",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
