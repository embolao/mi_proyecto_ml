# tests/test_example.py
from ml import __version__  # Asegúrate de que tu paquete sea importable


def test_version():
    assert __version__ is not None
