from pathlib import Path
import numpy as np
from typing import Tuple

def save_mnist_as_npy(data: Tuple, save_dir: str) -> None:
    """
    Guarda los datos MNIST en formato .npy
    
    Args:
        data: Tupla con ((train_images, train_labels), (test_images, test_labels))
        save_dir: Directorio donde guardar los archivos
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    (x_train, y_train), (x_test, y_test) = data
    
    np.save(save_path/'train_images.npy', x_train)
    np.save(save_path/'train_labels.npy', y_train)
    np.save(save_path/'test_images.npy', x_test)
    np.save(save_path/'test_labels.npy', y_test)

def load_mnist_from_npy(load_dir: str) -> Tuple:
    """
    Carga datos MNIST desde archivos .npy
    
    Args:
        load_dir: Directorio que contiene los archivos .npy
        
    Returns:
        Tupla con ((train_images, train_labels), (test_images, test_labels))
    """
    load_path = Path(load_dir)
    
    return (
        (np.load(load_path/'train_images.npy'), 
         np.load(load_path/'train_labels.npy')),
        (np.load(load_path/'test_images.npy'), 
         np.load(load_path/'test_labels.npy'))
    )