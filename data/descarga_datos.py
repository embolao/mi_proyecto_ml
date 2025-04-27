import os
from contextlib import redirect_stderr, redirect_stdout
import io
import numpy as np
from tensorflow import keras
from os.path import join

def descargar_mnist(ruta_descarga):
    """Descarga y guarda el dataset MNIST de forma silenciosa"""
    try:
        # Verificar si los archivos ya existen
        archivos_requeridos = [
            'train_images.npy',
            'train_labels.npy',
            'test_images.npy',
            'test_labels.npy'
        ]
        
        todos_existen = all(os.path.exists(join(ruta_descarga, f)) for f in archivos_requeridos)
        
        if todos_existen:
            print("\n✅ Los archivos de MNIST ya están creados en:", ruta_descarga)
            return True

        # Crear directorio si no existe
        os.makedirs(ruta_descarga, exist_ok=True)
        
        # Descargar dataset (con redirección temporal)
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Guardar los archivos
        np.save(join(ruta_descarga, 'train_images.npy'), x_train)
        np.save(join(ruta_descarga, 'train_labels.npy'), y_train)
        np.save(join(ruta_descarga, 'test_images.npy'), x_test)
        np.save(join(ruta_descarga, 'test_labels.npy'), y_test)
        
        print(f"\n✅ MNIST descargado correctamente en: {ruta_descarga}")
        print(f"• Conjunto de entrenamiento: {x_train.shape} ({x_train.dtype})")
        print(f"• Conjunto de prueba: {x_test.shape} ({x_test.dtype})")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error al descargar MNIST: {str(e)}")
        return False

if __name__ == "__main__":
    ruta = '/mnt/codigos/mi_proyecto1/data/mnist'
    
    if descargar_mnist(ruta):
        # Verificación detallada de archivos
        print("\nEstado de los archivos:")
        for archivo in ['train_images.npy', 'train_labels.npy', 'test_images.npy', 'test_labels.npy']:
            path = join(ruta, archivo)
            if os.path.exists(path):
                tamaño_mb = os.path.getsize(path)/(1024*1024)
                print(f"- {archivo}: {tamaño_mb:.2f} MB (Ya existía)" if "exist" in locals() else f"- {archivo}: {tamaño_mb:.2f} MB (Creado ahora)")
            else:
                print(f"- {archivo}: No se pudo crear")