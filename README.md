# Classificação de Tumores Mamários com CNN

Projeto desenvolvido para a disciplina de Tópicos de Matemática Aplicada (UFS) que implementa um classificador binário de tumores mamários em mamografias usando Redes Neurais Convolucionais (CNNs).

## Objetivo
Classificar imagens de mamografias em:
- **Classe 0**: Tumores benignos
- **Classe 1**: Tumores malignos

## Tecnologias
- Python 3.x
- TensorFlow/Keras
- OpenCV (pré-processamento)
- Scikit-learn (análise comparativa)

## Arquitetura da CNN
```python
model = tf.keras.Sequential([
    # Bloco 1: 32 filtros + maxpooling
    Conv2D(32, (3,3), activation='relu', input_shape=(222,222,1)),
    MaxPooling2D((2,2)),
    # Bloco 2: 64 filtros + maxpooling
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    # Bloco 3: 128 filtros + maxpooling
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    # Regularização
    Dropout(0.3),
    # Classificador
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])
