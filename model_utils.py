from tensorflow.keras import layers, models

IMAGE_SHAPE = (150, 150, 3)
N_CLASSES = 4


def build_cnn(base_filters=32, dropout=0.5):
    f = base_filters
    return models.Sequential([
        layers.Input(shape=IMAGE_SHAPE),
        layers.Conv2D(f, 4, activation='relu'),
        layers.MaxPooling2D(3),
        layers.Conv2D(2*f, 4, activation='relu'),
        layers.MaxPooling2D(3),
        layers.Conv2D(4*f, 4, activation='relu', name='gradcam_conv'),
        layers.MaxPooling2D(3),
        layers.Conv2D(4*f, 4, activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(N_CLASSES, activation='softmax'),
    ])