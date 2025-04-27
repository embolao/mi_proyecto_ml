import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

class MNISTVisualizer:
    """
    Herramientas para visualización de datos MNIST
    """
    
    @staticmethod
    def plot_samples(images: np.ndarray, 
                    labels: np.ndarray, 
                    num_samples: int = 10, 
                    title: str = "", 
                    indices: Optional[List[int]] = None) -> None:
        """
        Muestra una cuadrícula de muestras MNIST
        
        Args:
            images: Array de imágenes (N, 28, 28)
            labels: Array de etiquetas (N,)
            num_samples: Número de muestras a mostrar
            title: Título para el gráfico
            indices: Lista opcional de índices específicos a mostrar
        """
        if indices is None:
            if len(images) < num_samples:
                raise ValueError(f"Solo {len(images)} muestras disponibles, pero se solicitaron {num_samples}")
            indices = range(num_samples)
        
        plt.figure(figsize=(15, 6))
        for i, idx in enumerate(indices):
            plt.subplot(2, 5, i+1)
            plt.imshow(images[idx], cmap='gray')
            plt.title(f"Label: {labels[idx]}")
            plt.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_class_distribution(labels: np.ndarray, 
                               title: str = "Distribución de Clases") -> None:
        """
        Muestra un gráfico de barras con la distribución de clases
        
        Args:
            labels: Array de etiquetas
            title: Título del gráfico
        """
        unique, counts = np.unique(labels, return_counts=True)
        plt.figure(figsize=(10, 5))
        plt.bar(unique, counts)
        plt.title(title)
        plt.xlabel("Dígito")
        plt.ylabel("Cantidad")
        plt.xticks(unique)
        plt.grid(axis='y')
        plt.show()