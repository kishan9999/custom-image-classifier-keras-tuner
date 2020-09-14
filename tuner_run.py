# Libraries
import tensorflow as tf
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# DATA pre-process

# Importing/load files from Gdrive
data = pd.read_csv("dataset.csv")

# Random Suffle Data
data = data.sample(frac = 1) 

# Image Label Seperating
X = data.drop(['label'],1).values
Y = data['label'].values

# Visualize datasets
plt.imshow(X[500].reshape(28,28))
plt.show()

# Normlizing and Reshaping
X = X/255.
X = X.reshape(-1,28,28,1)

# Data Labeling (one hot enconding)
b = pd.get_dummies(Y)
b2 = b.values.argmax(1)
Y = to_categorical(b2, dtype='Int32')  

# Methodology (Keras-Tuner)

# This function returns a compiled model.
def model_definition(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=hp.Int('convolution',min_value=32, max_value=128, step=32),
                               kernel_size=hp.Choice('kernel',values =[3,5]),
                               activation='relu',input_shape=(28,28,1)),
        tf.keras.layers.Conv2D(filters=hp.Int('convolution',min_value=32, max_value=64, step=16),
                               kernel_size=hp.Choice('kernel',values =[3,5]),activation='relu'),
        tf.keras.layers.Conv2D(filters=hp.Int('convolution',min_value=16, max_value=32, step=8),
                               kernel_size=hp.Choice('kernel',values =[3,4]),activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=hp.Int('denselayer',min_value=16,max_value=64,step=16),
                              activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4])),
              loss='categorical_crossentropy',metrics=['accuracy'])
    return model

# Defining Tuner optimization method: You can try RandomSearch and Hyperband.
tuner = RandomSearch(model_definition, objective='val_accuracy',max_trials=8, 
                     executions_per_trial=2,directory='output',project_name='music_notes')

# View the summary of the search space:
tuner.search_space_summary()

# Finding the best hyperparameter configuration. which is similar to model.fit().
tuner.search(X, Y,epochs=10,validation_split=0.2, shuffle = True)

# Result Summary
tuner.results_summary()

# Best Models
best_models = tuner.get_best_models(num_models=2) 

# The Best Model Architecture
best_models[0].summary()

# Re training and improving The Best Model
best_models[0].fit(X, Y, epochs=20, validation_split=0.3, initial_epoch= 2, shuffle=True) 


