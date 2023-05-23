import tensorflow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

actions = [
    'swipe',
    'stop'
]

data = np.concatenate([
    # np.load('dataset/seq_come_1627646273.npy'),
    # np.load('dataset/seq_away_1627646273.npy'),
    # np.load('dataset/seq_spin_1627646273.npy')
    np.load('dataset/seq_swipe_1684800550.npy'),
    np.load('dataset/seq_stop_1684800550.npy'),

], axis=0)

print(data.shape)
x_data = data[:, :, :-1]
labels = data[:, 0, -1]

print(x_data.shape)
print(labels.shape) 
print(labels)

# for i in range(len(labels)) :
#     if labels[i] == 2 :
#         labels[i] = 1
         

y_data = to_categorical(labels, num_classes=len(actions))
y_data.shape

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[
        ModelCheckpoint('models/model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)
