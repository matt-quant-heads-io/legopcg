import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pandas as pd
import numpy as np
from keras.utils import np_utils
import random


from keras.layers import Input, Dense, Conv3D, Concatenate, MaxPooling3D, Flatten
from keras.optimizers import SGD
from keras.models import Model
import tensorflow as tf









# print(f"move_diff:\n\n\n{move_diff}")



# print(f"X.shape: {X.shape}")
# print(f"X.shape: {move_diff.shape}")



def batch_generator(Train_df, obs_size):
    return load_data(Train_df,obs_size)## Yields data
        


def load_data(Train_df,obs_size):

    dfs = []
    X = []
    y = []

    batch_size = 100000
    rows_per_df = 100000

    for file in os.listdir(Train_df):
        if not file.endswith('.csv'):
            continue
        print(f"compiling df {file}")
        # rand_idx = random.randint(batch_size, (rows_per_df-1))
        df = pd.read_csv(f"{Train_df}/{file}")
        dfs.append(df)

    df = pd.concat(dfs)

    df = df.sample(frac=1).reset_index(drop=True)
    y_true = df[['target']]
    y = np_utils.to_categorical(y_true)
    df.drop('target', axis=1, inplace=True)
    y = y.astype('int32')

    move_diff = df[['move_diff']]
    move_diff = np_utils.to_categorical(move_diff)
    df.drop('move_diff', axis=1, inplace=True)
    move_diff = move_diff.astype('int32')
    # move_diff = np.column_stack((move_diff))


    for idx in range(len(df)):
        x = df.iloc[idx, :].values.astype('int32')
        row = []
        for val in x:
            # print(f"val: {val}")
            oh = [0]*37
            oh[val] = 1
            row.append(oh)
        X.append(np.array(row).reshape((obs_size,obs_size,obs_size,37)))

    X = np.array(X)


    assert len(X) == len(move_diff), f"len(X): {len(X)}, len(move_diff): {len(move_diff)}"
    return [X, move_diff], y






obs_size = 6
data_size = 1
version = 3

pod_root_path = f'{os.path.dirname(os.path.abspath(__file__))}/data/trajectories/racers'
model_abs_path = f"{os.path.dirname(os.path.abspath(__file__))}/saved_models/racers_obs_{obs_size//2}.h5"

# dfs = []
# for file in os.listdir(pod_root_path):
#     if not file.endswith('.csv'):
#         continue
#     print(f"compiling df {file}")
#     df = pd.read_csv(f"{pod_root_path}/{file}", nrows=15000)
#     dfs.append(df)

# df = pd.concat(dfs)

# df = df.sample(frac=1).reset_index(drop=True)
# y_true = df[['target']]
# y = np_utils.to_categorical(y_true)
# print(f"y shape {y.shape}")

# input('')


inputs = [
    Input(shape=(obs_size, obs_size, obs_size, 37)),
    Input(shape=(3,)),
]

x = Conv3D(128, (3, 3, 3), activation='relu', input_shape=(obs_size, obs_size, obs_size, 37), padding="SAME")(inputs[0])

x = MaxPooling3D(pool_size=(2, 2, 2))(x)
x = Conv3D(128, (3, 3, 3), activation='relu', padding="SAME")(x)
x = Conv3D(256, (3, 3, 3), activation='relu', padding="SAME")(x)
x = Flatten()(x)
x = Concatenate()([x, inputs[1]])

output = Dense(37, activation="softmax")(x)

conditional_cnn_model = Model(inputs, output)



conditional_cnn_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=[tf.keras.metrics.CategoricalAccuracy()])
mcp_save = ModelCheckpoint(model_abs_path, save_best_only=True, monitor='categorical_accuracy', mode='max')


for i in range(20):
    
    [X,move_diff], targets = batch_generator(f"{os.path.dirname(os.path.abspath(__file__))}/data/trajectories/racers", obs_size)
    print(f"train_data: {[X,move_diff]}")
    print(f"targets: {targets}")
    history = conditional_cnn_model.fit([X,move_diff],targets, epochs=10, steps_per_epoch=64, verbose=2, callbacks=[mcp_save])
    X = None
    move_diff = None
    targets = None

# conditional_cnn_model.compile([X,move_diff]
#     loss="categorical_crossentropy",
#     optimizer=SGD(),
#     metrics=[tf.keras.metrics.CategoricalAccuracy()],
# )





# model = tf.keras.models.Sequential([
#     tf.keras.layers.,
#     tf.keras.layers.,
#     tf.keras.layers.,
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', input_shape=(obs_size, obs_size, obs_size, 38), padding="SAME"),
#     tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
#     tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding="SAME"),
#     tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding="SAME"),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])




# history = conditional_cnn_model.fit([X, move_diff], y, epochs=500, steps_per_epoch=64, verbose=2, callbacks=[mcp_save])
# history = conditional_cnn_model.fit_generator(generator=train_gen, epochs=500, steps_per_epoch=64, verbose=2, callbacks=[mcp_save])