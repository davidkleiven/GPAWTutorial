import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import save_model, load_model
import numpy as np
from matplotlib import pyplot as plt
import h5py as h5
tf.logging.set_verbosity(tf.logging.INFO)
print(tf.__version__)


class PrintDot(tf.keras.callbacks.Callback):
    def __init__(self, *args, **kwargs):
        super(PrintDot, self).__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
            print("Epoch: {}.".format(epoch))
        print('.', end='')

    def on_train_batch_begin(self, batch, logs):
        pass

    def on_train_batch_end(self, batch, logs):
        pass

def concentration_activated_nn(params, show=True, fromfile=None):
    X = np.loadtxt("data/cf_matrix.csv", delimiter=",")
    y = np.loadtxt("data/e_dft.csv", delimiter=",")
    y -= np.mean(y)
    X -= np.mean(X, axis=0)

    droprate = params.get('droprate', None)


    N = 1
    num_elements = 4

    # Start with a Conv1D such that we can sparsify the input
    input_layer = layers.Input((X.shape[1],))

    num_first_layer = 4
    l1_layer = layers.Dense(num_first_layer, activation="relu", kernel_regularizer=regularizers.l1(params['alpha']))(input_layer)

    if droprate is not None:
        l1_layer = layers.Dropout(droprate)(l1_layer)

    # CE model layer (with dropout)
    num_ce_models = params['layers'][0]
    hidden = layers.Dense(num_ce_models)(l1_layer)
    if droprate is not None:
        hidden = layers.Dropout(droprate)(hidden)
    
    for i in range(1, len(params['layers'])):
        hidden = layers.Dense(params['layers'][i], activation="relu")(hidden)
        if droprate is not None:
            hidden = layers.Dropout(droprate)(hidden)

    out = layers.Dense(1, activation="linear")(hidden)

    if fromfile is None:
        full_model = Model(inputs=input_layer, outputs=out)

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        full_model.compile(loss='mean_squared_error',
                    optimizer=optimizer,
                    metrics=['mean_absolute_error', 'mean_squared_error'])
    else:
        with h5.File(fromfile, mode='r') as hf:
            full_model = load_model(hf)
        print("Model loaded from {}".format(fromfile))
        
    train_X = X[:-30, :]
    train_y = y[:-30]
    test_X = X[-30:, :]
    y_test = y[-30:]

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        train_X, train_y, every_n_steps=50)

    # Create the input
    history = full_model.fit(
        train_X, train_y,
        epochs=1000000, validation_split=0.2, verbose=0,
        callbacks=[PrintDot()])

    filepath = "data/tf_model"
    fname = "data/tf_model.h5"
    with h5.File(fname, mode='w') as hfile:
        save_model(full_model, hfile, overwrite=True, include_optimizer=True)
    print("H5 model save to {}".format(fname))

    predicted_train = full_model.predict(train_X)[:, 0]
    predict_test = full_model.predict(test_X)[:, 0]

    mse_train = np.sqrt(np.mean((predicted_train - train_y)**2))
    mse_test = np.sqrt(np.mean((predict_test - y_test)**2))
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(history.epoch, history.history["mean_absolute_error"])
        ax.plot(history.epoch, history.history["val_mean_absolute_error"])
        ax.set_yscale("log")
        ax.set_xscale("log")

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1, 1, 1)
        
        print(predicted_train.shape)
        print()
        ax2.plot(predicted_train - train_y, 'o', mfc="none")
        ax2.plot(predict_test - y_test, 'x')
        fig.savefig("data/progress.png")
        fig2.savefig('data/residuals.png')
        plt.show()
    print("Mse train: {}".format(mse_train))
    print("Mse test: {}".format(mse_test))
    return mse_train, mse_test


def main():
    X = np.loadtxt("data/cf_matrix.csv", delimiter=",")
    y = np.loadtxt("data/e_dft.csv", delimiter=",")

    model = Sequential([
        layers.Dense(4, activation="linear", input_shape=[X.shape[1]]),
        layers.Dense(4, activation="relu"),
        layers.Dense(1, activation="linear")
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    
    train_X = X[:-10, :]
    train_y = y[:-10]

    history = model.fit(
        train_X, train_y,
        epochs=5000, validation_split=0.2, verbose=0,
        callbacks=[PrintDot()])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(history.epoch, history.history["mean_absolute_error"])
    ax.plot(history.epoch, history.history["val_mean_absolute_error"])
    fig.savefig("data/progress.png")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(train_y, model.predict(train_X), 'o')
    ax2.plot(train_y, train_y, 'x')
    fig2.savefig('data/residuals.png')
    plt.show()

concentration_activated_nn(show=True, params={'alpha': 1E-6, 'layers': [4, 4, 4]})#, fromfile="data/tf_model.h5")
#main()