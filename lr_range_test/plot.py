import numpy as np
import matplotlib.pyplot as plt
from .utils import add_time


def plot_metrics(history, acc='accuracy', loss='loss', 
                 val_acc='val_accuracy', val_loss='val_loss'):
    acc      = history.history[acc]
    val_acc  = history.history[val_acc]
    loss     = history.history[loss]
    val_loss = history.history[val_loss]
    epochs   = range(len(acc))
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def plot_lr_range_test_from_hist(history,
                                filename = "learn_rate_range_test",
                                max_loss = 5,
                                max_lr = 1):
    loss = np.asarray(history.history['loss'])
    lr   = np.asarray(history.history['lr'])
    cut_index = np.argmax(loss > max_loss)
    if cut_index == 0:
        print("\nLoss did not exceed `MAX_LOSS`.")
        print("Increase `epochs` and `MAX_LR`, or decrease `MAX_LOSS`.")
        print("\nPlotting with full history. May be scaled incorrectly...\n\n")
    else:
        loss[cut_index] = max_loss
        loss = loss[:cut_index]
        lr = lr[:cut_index]
    lr_cut_index = np.argmax(lr > max_lr)
    if lr_cut_index != 0:
        lr[lr_cut_index] = max_lr
        lr = lr[:lr_cut_index]
        loss = loss[:lr_cut_index]
    # Plot (TODO: try and use ggplot2 here with module to call R?)
    ploty(loss, lr, xlab='Learning Rate', ylab='Loss', filepath='lr_range_plot')

def ploty(y, x=None, xlab='obs', ylab='value', 
          save=True, title='', filepath='plot'):
    if x is None: x = np.linspace(0, len(y), len(y))
    filepath = add_time(filepath)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel=xlab, ylabel=ylab, title=title)
    ax.grid()
    if save:
       fig.savefig(filepath)
    plt.show()
    return filepath