#Some helpers

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style('whitegrid')



def open_protocol(project_name):
    document_name = project_name + ".xlsx"
    try:
        protocol = pd.read_excel(document_name)
        current_model_no = str(protocol['model_no'].iloc[-1] + 1)

    except:
        protocol = pd.DataFrame(columns=['model_no','train_size','val_size','test_size' ,'epochs', 'batch_size',
                                 'learning_rate', 'momentum', 'test_acc', 'graphic'])
        writer = pd.ExcelWriter(document_name, engine='xlsxwriter')
        protocol.to_excel(writer, sheet_name='protocol', index = False)
        writer.save()
        current_model_no = "1"
    return document_name, current_model_no

def save_protocol(document_name, current_model_no, train_size, val_size, test_size, epochs, batch_size, learning_rate,
                  momentum, test_acc, graphic):
    protocol = pd.read_excel(document_name)
    graphic = graphic + '.jpg'
    protocol.loc[protocol['model_no'].max()] = [current_model_no, train_size, val_size, test_size, epochs, batch_size,
                                         learning_rate, momentum, test_acc, graphic]
    writer = pd.ExcelWriter(document_name, engine='xlsxwriter')
    protocol.to_excel(writer, sheet_name='protocol', index=False)
    writer.save()


def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    predictions = predictions.cpu()
    y = y.cpu()
    return np.mean(np.equal(predictions.numpy(), y.numpy()))

def create_plot(train_lossd, val_lossd, train_accd, val_accd, epochs, fig_name):

    fig,(ax0, ax1) = plt.subplots(2)

    fig.dpi = 300

    ax0.plot(*zip(*train_accd.items()), label = 'training accuracy')
    ax0.plot(*zip(*val_accd.items()), label = 'validation accuracy')
    ax0.legend()
    ax0.set_title(fig_name)


    ax1.plot(*zip(*train_lossd.items()), label= 'training loss')
    ax1.plot(*zip(*val_lossd.items()), label = 'validation loss')
    ax1.legend()

    ax1.set_xlabel("Epoch")
    ax1.set_xticks(np.arange(1, epochs + 1, 5))

    for ax in fig.get_axes():
        ax.label_outer()

    fig_name='figures/' + fig_name + '.jpg'

    fig.savefig(fig_name)
    plt.show()


def train(train_data, val_data, model, epochs, lr, momentum, fig_name):
    optimizer = optim.sgd(model.parameters(), lr=lr, momentum=momentum)
    #optimizer = optim.Adam(model.parameters(), lr=lr)

    train_lossd = {}
    val_lossd = {}
    train_accd = {}
    val_accd = {}
    model_name=fig_name + '.pth'


    for epoch in range(1, epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))

        # Training
        loss, acc = run_epoch(train_data, model.train(), optimizer)
        print('Train loss: {:.6f} | Train accuracy: {:.6f}'.format(loss, acc))
        train_lossd[epoch] = loss
        train_accd[epoch] = acc


        # Validation
        val_loss, val_acc = run_epoch(val_data, model.eval(), optimizer)
        print('Val loss:   {:.6f} | Val accuracy:   {:.6f}'.format(val_loss, val_acc))
        val_lossd[epoch] = val_loss
        val_accd[epoch] = val_acc

        # Save Model
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, model_name)

    create_plot(train_lossd, val_lossd,train_accd,val_accd, epochs=epochs, fig_name=fig_name)
    return val_acc


def run_epoch(data, model, optimizer):
    """Train model for one pass of train data, and return loss, acccuracy"""
    # Gather losses
    losses = []
    batch_accuracies = []

    # If model is in train mode, use optimizer.
    is_training = model.training

    # Iterate through batches
    for x, y in (tqdm(data)):
        x = Variable(x.cuda())
        y = Variable(y.cuda())

        # Get output predictions
        out = model(x)
        model = model.cuda()

        # Predict and store accuracy
        predictions = torch.argmax(out, dim=1)
        predictions = Variable(predictions.cuda())
        batch_accuracies.append(compute_accuracy(predictions, y))

        # Compute loss
        loss = F.cross_entropy(out, y)
        losses.append(loss.data.item())

        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(batch_accuracies)
    return avg_loss, avg_accuracy