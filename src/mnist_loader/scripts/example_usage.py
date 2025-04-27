import os
from dotenv import load_dotenv
from mnist_loader import MNISTLoader, MNISTVisualizer, MNISTPreprocessor
from mnist_loader.utils import load_mnist_from_npy

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

def main():
    # Cargar rutas desde las variables de entorno
    data_path = os.getenv('DATA_PATH', '/ruta/por/default/mnist')  # Usa ruta por defecto si no está definida
    save_path = os.getenv('SAVE_PATH', '/ruta/por/default/processed')  # Ruta de salida por defecto

    print("Cargando datos desde:", data_path)
    (x_train, y_train), (x_test, y_test) = load_mnist_from_npy(data_path)
    
    print("Visualizando muestras...")
    MNISTVisualizer.plot_samples(x_train, y_train, title="Training Samples")
    MNISTVisualizer.plot_class_distribution(y_train)
    
    print("Normalizando datos...")
    x_train = MNISTPreprocessor.normalize(x_train)
    x_test = MNISTPreprocessor.normalize(x_test)
    
    print(f"Guardando datos preprocesados en: {save_path}")
    MNISTPreprocessor.save_processed_data(
        (x_train, y_train, x_test, y_test),
        save_dir=save_path
    )
    
    print("¡Proceso completado exitosamente!")

if __name__ == "__main__":
    main()
