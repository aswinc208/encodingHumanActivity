import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
import scipy.stats as st
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout, Conv2D, Lambda, Input, Bidirectional, Flatten
from tensorflow.keras import optimizers, backend as K
import matplotlib.pyplot as plt

def model(x_train, num_labels, LSTM_units, dropout, num_conv_filters, batch_size):
    inputs = Input(shape=(x_train.shape[1], x_train.shape[2]))
    x = Conv2D(num_conv_filters, (3, 3), activation='relu')(inputs)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(LSTM_units, activation='relu')(x)
    x = Dropout(dropout)(x)
    output = Dense(num_labels, activation='softmax')(x)
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_true, y_pred_classes)
    recall = recall_score(y_true, y_pred_classes, average='macro')
    f1 = f1_score(y_true, y_pred_classes, average='macro')
    cm = confusion_matrix(y_true, y_pred_classes)
    return accuracy, recall, f1, cm

def plot_training_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'ro-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_sample_data(X, y, sample_indices, num_samples=5):
    plt.figure(figsize=(15, 10))
    for i, index in enumerate(sample_indices[:num_samples]):
        plt.subplot(num_samples, 1, i + 1)
        plt.plot(X[index])
        plt.title(f'Sample {index} - Label: {y[index]}')
        plt.xlabel('Time Steps')
        plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

sample_indices = [0, 1, 2, 3, 4]
plot_sample_data(X, y, sample_indices)
