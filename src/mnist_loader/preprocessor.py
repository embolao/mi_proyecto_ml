import numpy as np
from typing import Tuple

class MNISTPreprocessor:
    """
    Funciones de preprocesamiento para datos MNIST
    """
    
    @staticmethod
    def normalize(images: np.ndarray) -> np.ndarray:
        """
        Normaliza imágenes al rango [0, 1]
        
        Args:
            images: Array de imágenes (N, 28, 28)
            
        Returns:
            Array normalizado
        """
        return images / 255.0
    
    @staticmethod
    def reshape_for_mlp(images: np.ndarray) -> np.ndarray:
        """
        Redimensiona imágenes para modelos MLP (aplanamiento)
        
        Args:
            images: Array de imágenes (N, 28, 28)
            
        Returns:
            Array redimensionado (N, 784)
        """
        return images.reshape(images.shape[0], -1)
    
    @staticmethod
    def reshape_for_cnn(images: np.ndarray) -> np.ndarray:
        """
        Redimensiona imágenes para modelos CNN (agrega canal)
        
        Args:
            images: Array de imágenes (N, 28, 28)
            
        Returns:
            Array redimensionado (N, 28, 28, 1)
        """
        return images[..., np.newaxis]
    
    @staticmethod
    def train_test_split(data: Tuple, 
                        test_size: float = 0.2, 
                        random_seed: int = None) -> Tuple:
        """
        Divide los datos en conjuntos de entrenamiento y validación
        
        Args:
            data: Tupla con (images, labels)
            test_size: Proporción para validación
            random_seed: Semilla para reproducibilidad
            
        Returns:
            Tupla con (X_train, X_val, y_train, y_val)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        images, labels = data
        num_val = int(len(images) * test_size)
        indices = np.random.permutation(len(images))
        
        val_idx = indices[:num_val]
        train_idx = indices[num_val:]
        
        return (images[train_idx], images[val_idx], 
                labels[train_idx], labels[val_idx])
    @staticmethod
    def save_processed_data(data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                             save_dir: str,
                             filename: str = "mnist_processed.npz") -> None:
        """
        Guarda los datos procesados en un único archivo .npz comprimido.
        Si el archivo ya existe, crea una nueva versión usando fecha y hora (YYYYMMDD_HHMM).
        
        Args:
            data: Tupla (x_train, y_train, x_test, y_test)
            save_dir: Directorio donde guardar el archivo
            filename: Nombre base del archivo .npz
        """
        import os
        import time
        
        x_train, y_train, x_test, y_test = data
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        base_name, ext = os.path.splitext(filename)
        
        # Obtener la fecha y hora actual
        timestamp = time.strftime("%Y%m%d_%H%M")
        new_filename = f"{base_name}_{timestamp}{ext}"
        save_path = os.path.join(save_dir, new_filename)
        
        np.savez_compressed(save_path,
                            x_train=x_train,
                            y_train=y_train,
                            x_test=x_test,
                            y_test=y_test)
        
        print(f"✅ Datos guardados exitosamente en: {save_path}")
