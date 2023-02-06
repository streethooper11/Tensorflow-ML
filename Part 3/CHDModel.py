#Original Author: Jonathan Hudson
#CPSC 501 F22

# Modified from notMNIST-Complete.py and main3.py

import numpy as np
import pandas as pd

import tensorflow as tf
import sklearn.model_selection

tf.random.set_seed(1234)

print("--Get data--")
df = pd.read_csv('heart.csv')
(x_train_with_rows, x_test_with_rows, y_train, y_test) = sklearn.model_selection.train_test_split(
    df.loc[:, 'row.names':'age'],  # row names + input variables
    df['chd'],  # output variables
    random_state=0,
    train_size=0.75
)  # Training data gets 75%

# Save training data and test data to separate files, with row names, input variables, and output variable
pd.concat([x_train_with_rows, y_train], axis=1).to_csv('heart_train.csv', index=False)
pd.concat([x_test_with_rows, y_test], axis=1).to_csv('heart_test.csv', index=False)

x_train = x_train_with_rows.loc[:, 'sbp':]  # Exclude row.names from data
x_test = x_test_with_rows.loc[:, 'sbp':]

# Source: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.GroupBy.ngroup.html
# Convert text column to number by assigning a group number for each unique text value
x_train['famhist'] = x_train.groupby('famhist').ngroup()
x_test['famhist'] = x_test.groupby('famhist').ngroup()


print("--Make model--")
input_shape = (9, 1)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dropout(0.25, input_shape=input_shape),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("--Fit model--")
model.fit(x_train, y_train, epochs=100, verbose=2)

print("--Evaluate model--")
model_loss1, model_acc1 = model.evaluate(x_train,  y_train, verbose=2)
model_loss2, model_acc2 = model.evaluate(x_test,  y_test, verbose=2)
print(f"Train / Test Accuracy: {model_acc1*100:.1f}% / {model_acc2*100:.1f}%")
