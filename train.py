import tensorflow as tf
from tensorflow import keras

class ModelTrainer:
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            keras.layers.Dense(10, activation='softmax')
        ])

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.compile_model()
