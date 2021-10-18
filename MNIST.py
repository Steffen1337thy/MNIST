import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import random_split
from utils import *

def main():
    # Name of the Project for organisational reasons
    project_name= 'MNIST'
    document_name, current_model_no = open_protocol(project_name=project_name)
    fig_name = project_name + '_model_no_' + current_model_no

    # Parameters:
    epochs = 30
    batch_size = 64
    lr = 0.01
    momentum = 0.90


    # loading the train data
    train_data = datasets.MNIST('data', train=True,download = False,transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,),(0.3081,))]))

    # set size of train and validation set
    dataset_size = len(train_data)
    train_size = int(0.9*dataset_size)
    val_size = int(dataset_size - train_size)
    test_size = len(datasets.MNIST('data', train=False))

    # train/val - split

    train_set, val_set = random_split(train_data, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # putting data in dataloader
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_data = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle= True, **kwargs)

    val_data = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle= True, **kwargs)

    # also load test data
    test_data = torch.utils.data.DataLoader(datasets.MNIST('data', train=False,
                                         transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,),(0.3081,))])), batch_size=batch_size, shuffle= True, **kwargs)




    # the nn model
    class Netz(nn.Module):
        def __init__(self):
            super(Netz, self).__init__()
            self.conv1 = nn.Conv2d(1,32, kernel_size=5)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
            self.conv_dropout = nn.Dropout2d()
            self.fc1 = nn.Linear(1024, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.max_pool2d(x, 2)
            x = F.relu(x)
            x = self.conv2(x)
            x = self.conv_dropout(x)
            x = F.max_pool2d(x, 2)
            x = F.relu(x)
            #print(x.size())
            #exit()
            x = x.view(-1,1024)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim = 1)

    model = Netz()
    model.cuda()

    train(train_data, val_data, model, epochs=epochs, lr=lr, momentum=momentum, fig_name=fig_name)

    loss, accuracy = run_epoch(test_data, model.eval(), None)

    #print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))
    print('Loss on test set:   {:.6f} |  Accuracy on test set:   {:.6f}'.format(loss, accuracy))

    save_protocol(document_name=document_name, current_model_no=current_model_no,train_size=train_size,val_size=val_size,
                  test_size=test_size,epochs=epochs,batch_size=batch_size,learning_rate=lr,momentum=momentum,test_acc=accuracy, graphic=fig_name)

if __name__ == '__main__':
    main()
