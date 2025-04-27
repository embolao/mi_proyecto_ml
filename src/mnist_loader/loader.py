import numpy as np
import struct
from pathlib import Path
from typing import Tuple, Optional
import warnings

class MNISTLoader:
    def __init__(self, data_dir: str, use_npy: bool = False, auto_load: bool = True):
        """
        Inicializa el cargador de datos MNIST con capacidades mejoradas.
        
        Args:
            data_dir: Directorio que contiene los archivos MNIST
            use_npy: Si True, carga archivos .npy en lugar de los archivos IDX originales
            auto_load: Si True, carga los datos automáticamente al inicializar
        """
        self.data_dir = Path(data_dir)
        self.use_npy = use_npy
        self._train_images = None
        self._train_labels = None
        self._test_images = None
        self._test_labels = None
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"El directorio no existe: {self.data_dir}")
            
        if auto_load:
            self.load_data()

    def _validate_files(self, extension: str) -> None:
        """
        Valida la existencia de los archivos requeridos.
        
        Args:
            extension: Extensión de archivo a validar ('idx-ubyte' o 'npy')
        """
        if extension == 'idx-ubyte':
            files = [
                'train-images-idx3-ubyte',
                'train-labels-idx1-ubyte',
                't10k-images-idx3-ubyte',
                't10k-labels-idx1-ubyte'
            ]
        else:
            files = [
                'train_images.npy',
                'train_labels.npy',
                'test_images.npy',
                'test_labels.npy'
            ]
        
        missing = [f for f in files if not (self.data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Archivos {extension} no encontrados: {', '.join(missing)}"
            )

    @staticmethod
    def _load_idx(filepath: Path) -> np.ndarray:
        """
        Carga un archivo en formato IDX con validación robusta.
        """
        try:
            with open(filepath, 'rb') as f:
                magic, = struct.unpack('>I', f.read(4))
                
                if magic == 2049:  # Labels
                    size, = struct.unpack('>I', f.read(4))
                    data = np.frombuffer(f.read(), dtype=np.uint8)
                    if len(data) != size:
                        warnings.warn(f"Tamaño de etiquetas no coincide: esperado {size}, obtenido {len(data)}")
                    return data
                    
                elif magic == 2051:  # Images
                    size, rows, cols = struct.unpack('>III', f.read(12))
                    data = np.frombuffer(f.read(), dtype=np.uint8)
                    expected_size = size * rows * cols
                    if len(data) != expected_size:
                        raise ValueError(
                            f"Tamaño de imagen no coincide: esperado {expected_size}, obtenido {len(data)}"
                        )
                    return data.reshape(size, rows, cols)
                    
                raise ValueError(f"Formato IDX no reconocido: {magic}")
        except Exception as e:
            raise IOError(f"Error al leer {filepath}: {str(e)}")

    def load_data(self) -> None:
        """Carga todos los datos MNIST en memoria."""
        try:
            if self.use_npy:
                self._validate_files('npy')
                self._train_images = np.load(self.data_dir/'train_images.npy')
                self._train_labels = np.load(self.data_dir/'train_labels.npy')
                self._test_images = np.load(self.data_dir/'test_images.npy')
                self._test_labels = np.load(self.data_dir/'test_labels.npy')
            else:
                self._validate_files('idx-ubyte')
                self._train_images = self._load_idx(self.data_dir/'train-images-idx3-ubyte')
                self._train_labels = self._load_idx(self.data_dir/'train-labels-idx1-ubyte')
                self._test_images = self._load_idx(self.data_dir/'t10k-images-idx3-ubyte')
                self._test_labels = self._load_idx(self.data_dir/'t10k-labels-idx1-ubyte')
        except Exception as e:
            self._clear_data()
            raise

    def _clear_data(self) -> None:
        """Limpia los datos cargados."""
        self._train_images = None
        self._train_labels = None
        self._test_images = None
        self._test_labels = None

    def save_as_npy(self, save_dir: str, overwrite: bool = False) -> None:
        """
        Guarda los datos cargados en formato .npy.
        
        Args:
            save_dir: Directorio de destino
            overwrite: Si True, sobrescribe archivos existentes
        """
        if not self.is_loaded:
            raise ValueError("No hay datos cargados para guardar")
            
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        files = {
            'train_images.npy': self._train_images,
            'train_labels.npy': self._train_labels,
            'test_images.npy': self._test_images,
            'test_labels.npy': self._test_labels
        }
        
        for filename, data in files.items():
            filepath = save_path / filename
            if filepath.exists() and not overwrite:
                raise FileExistsError(f"El archivo ya existe: {filepath}")
            np.save(filepath, data)

    @property
    def is_loaded(self) -> bool:
        """Indica si los datos están cargados en memoria."""
        return all(
            x is not None 
            for x in [
                self._train_images, 
                self._train_labels,
                self._test_images,
                self._test_labels
            ]
        )

    @property
    def train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve (imágenes de entrenamiento, etiquetas)"""
        if not self.is_loaded:
            raise ValueError("Los datos no han sido cargados")
        return self._train_images, self._train_labels

    @property
    def test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve (imágenes de prueba, etiquetas)"""
        if not self.is_loaded:
            raise ValueError("Los datos no han sido cargados")
        return self._test_images, self._test_labels

    @property
    def shape_info(self) -> dict:
        """Devuelve información de dimensiones de los datos."""
        if not self.is_loaded:
            return {"status": "Datos no cargados"}
            
        return {
            "train_images": self._train_images.shape,
            "train_labels": self._train_labels.shape,
            "test_images": self._test_images.shape,
            "test_labels": self._test_labels.shape
        }