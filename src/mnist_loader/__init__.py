"""
Paquete para carga y procesamiento de datos MNIST

Módulos:
- loader: Carga de datos desde archivos IDX
- visualizer: Herramientas de visualización
- preprocessor: Funciones de preprocesamiento
- utils: Funciones auxiliares
"""

from .loader import MNISTLoader
from .visualizer import MNISTVisualizer
from .preprocessor import MNISTPreprocessor
from .utils import save_mnist_as_npy, load_mnist_from_npy

__all__ = [
    'MNISTLoader',
    'MNISTVisualizer',
    'MNISTPreprocessor',
    'save_mnist_as_npy',
    'load_mnist_from_npy'
]

__version__ = '0.1.0'