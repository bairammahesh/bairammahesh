import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D as Convolution2D
from tensorflow.keras.models import Sequential
import pickle
import os

# Load X and Y
X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')
X = X.astype('float32')
X = X / 255
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

print("Data loaded, shape:", X.shape, Y.shape)

# Train model
classifier = Sequential()
classifier.add(Convolution2D(32, (3, 3), input_shape=(46, 46, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(256, activation='relu'))
classifier.add(Dense(8, activation='softmax'))

print(classifier.summary())
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = classifier.fit(X, Y, batch_size=16, epochs=50, shuffle=True, verbose=2)

# Save
classifier.save('model/model.h5')
f = open('model/history.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()

acc = hist.history['accuracy']
accuracy = acc[-1] * 100
print("Training completed, final accuracy:", accuracy)
