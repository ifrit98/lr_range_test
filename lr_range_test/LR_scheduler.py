# import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from .utils import mnist_data, mnist_model
from .plot import plot_lr_range_test_from_hist, plot_metrics, ploty


def learn_rate_range_test(model, ds, init_lr=1e-4, factor=3, plot=True,
                          max_lr=1, max_loss=2, epochs=25, save_hist=True, verbose=1):

    lr_range_callback = tf.keras.callbacks.LearningRateScheduler(
        schedule = lambda epoch: init_lr * tf.pow(
            tf.pow(max_lr / init_lr, 1 / (epochs - 1)), epoch))
    hist = model.fit(
        ds,
        epochs = epochs,
        callbacks = [lr_range_callback],
        verbose=verbose)
    if save_hist:
        from pickle import dump
        f = open("lr-range-test-history", 'wb')
        dump(hist.history, f)
        f.close()
    if plot:
        plot_lr_range_test_from_hist(hist, max_lr=max_lr, max_loss=max_loss)

    return infer_best_lr_params(hist, factor)

def infer_best_lr_params(history, factor=3): 
    idx = tf.argmin(history.history['loss'])
    best_run_lr = history.history['lr'][idx]
    min_lr = best_run_lr / factor
    return [min_lr, best_run_lr]

class add_lr_to_history_obj(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if 'lr' not in self.model.history.history.keys():
            self.model.history.history['lr'] = list()
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr)
        self.model.history.history['lr'].append(lr)

def demo(epochs=25, max_lr=3):
    print("\n\nLOADING AND PROCESSING MNIST DATA...")
    (ds_train, ds_test) = mnist_data()

    print("LOADING SMALL MNIST MODEL...")
    model = mnist_model()

    print("CONDUCTING LEARNING_RATE RANGE TEST...")
    (min_lr, max_lr) = learn_rate_range_test(
        model, ds_train, max_lr=max_lr, epochs=epochs)
    print("Minimum learning rate:", min_lr)
    print("Maximum learning rate:", max_lr)

    print("\nRECOMPILING MODEL WITH NEW LEARNING RATE PARAM...")
    model = mnist_model(max_lr)
    print("FITTING NEW MODEL TO DATA...")
    h = model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
        verbose=1
    )
    print("PLOTTING METRICS WITH NEW LR...")
    plot_metrics(h)
    return {
        'history': h, 
        'min_max_lr': (min_lr, max_lr)
    }
    # TODO Show improvement/differences between baseline and new runs?