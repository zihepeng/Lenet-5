import os, random, shutil
from os import walk
import numpy as np
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
from model import Net


def make_directory(train_data_dir,new_dir):
    """Define the data directory
    """
    directories=os.listdir(train_data_dir)
    for directory in directories:
        for c in ['train','valid']:
            path = new_dir + '/' + c + '/' + directory
            if not os.path.exists(path):
                os.makedirs(path)


def divide_train_validation(train_data_dir,new_dir):
    """Get the data into corresponding directory
    """
    class_names = os.listdir(train_data_dir)
    for classes in class_names:
        name = os.listdir(os.path.join(train_data_dir, classes))
        random.shuffle(name)
        training = name[0:int(0.9 * len(name))]
        validation = name[int(0.9 * len(name)):]
        for pic in training:
            shutil.copyfile(train_data_dir + '/' + classes + '/' + pic, new_dir+ '/train/' + classes + '/' + pic)
        for pic in validation:
            shutil.copyfile(train_data_dir + '/' + classes + '/' + pic, new_dir+ '/valid/' + classes + '/' + pic)


def get_data(data_dir,c):
    """Transform the data in DataLoader format

    Arguments:
        data_dir (str):     The directory of training data

    Return:
        data (DataLoader):  The data in DataLoader format
    """
    if c=='val':
        transform = transforms.Compose([transforms.RandomCrop(200),
                                    transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
        data = datasets.ImageFolder(data_dir, transform)
    if c=='train':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(200),
                                    transforms.ToTensor(),
                                 transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254/255, 0, 0)),
                                   transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
        data = datasets.ImageFolder(data_dir, transform)
    return DataLoader(dataset=data, batch_size=16, shuffle=True)


def val(model,test_dir):
    """Test model performance on validation dataset.

    Arguments:
        train_data_dir (str):   The directory of training data
        model_dir (str):        The directory of the saved model.

    Return:
        validation_accuracy (float): The training accuracy.
    """
    test_loader = get_data(test_dir,'val')
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    loss_fn = nn.NLLLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data,target = Variable(data),Variable(target)
            output = model(data)
            loss = loss_fn(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        avg_loss = test_loss / len(test_loader)
        print('Validation Dataset')
        print('Average Loss:{:.4f} Accuracy:{:.4f}\n'.format(avg_loss,acc))
    return acc


def train(train_dir, model_dir):
    """Main training model.

    Arguments:
        train_data_dir (str):   The directory of training data
        model_dir (str):        The directory of the saved model.

    Return:
        train_accuracy (float): The training accuracy.
    """
    real_dir = './data_train_valid'
    make_directory(train_dir, real_dir )
    divide_train_validation(train_dir, real_dir )
    train_loader = get_data(real_dir+'/train','train')
    model = Net()
    best_val_acc=0
    final_train_acc=0
    loss_fn = nn.NLLLoss()
    for epoch in range(100):
        optimizer = torch.optim.Adam(model.parameters(),lr = 0.001* (0.1 ** (epoch // 30)),weight_decay=0)
        print('************** Epoch {} **************'.format(epoch + 1))
        total_loss = 0
        correct = 0
        total_num = len(train_loader.dataset)
        for batch_id, (data, target) in enumerate(train_loader):
            data,target = Variable(data),Variable(target)
            pred = model(data)
            loss = loss_fn(pred,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print_loss = loss.data.item()
            total_loss += print_loss
            _, pred = torch.max(pred.data, 1)
            correct += torch.sum(pred == target)
            # print training information
            if (batch_id + 1) % 30 == 0:
                print('[{}/{} ({:.0f}%)]\tLoss:{:.4f}'.format((batch_id + 1) * len(data), len(train_loader.dataset),
                                                              100. * (batch_id + 1) / len(train_loader), loss.item()))
                
        # calculate average loss
        avg_loss = total_loss / len(train_loader)
        # calculate accuracy
        correct = correct.data.item()
        acc = correct / total_num
        print('\nTraining Dataset')
        print('Average Loss:{:.4f} Accuracy:{:.4f}'.format(avg_loss, acc))
        # check the performance on validation dataset
        vcc = val(model,real_dir+'/valid')
        if vcc > best_val_acc:
            best_val_acc = vcc
            final_train_acc = acc
            torch.save(model, model_dir)
    return final_train_acc
            

def test(test_data_dir, model_dir):
    """Main testing model.

    Arguments:
        test_data_dir (str):    The `test_data_dir` is blind to you. But this directory will have the same folder structure as the `train_data_dir`.
                                You could reuse the snippets of loading data in `train` function
        model_dir (str):        The directory of the saved model. You should load your pretrained model for testing
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        test_accuracy (float): The testing accuracy.
    """
    transform = transforms.Compose([transforms.RandomCrop(200),
                                    transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    dataset_test = datasets.ImageFolder(test_data_dir, transform)
    test_loader =DataLoader(dataset=dataset_test, batch_size=16,shuffle=True)
    model = torch.load(model_dir)
    model.eval()
    loss_fn = nn.NLLLoss()
    correct = 0
    total_num = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            output = model(data)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
        correct = correct.data.item()
        acc = correct / total_num
        #print('Accuracy: {}/{} {:.4f}'.format(correct, len(test_loader.dataset), acc))
    return acc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train','test'])
    parser.add_argument('--train_data_dir', default='./data/train/', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./data/test/', help='the directory of testing data')
    parser.add_argument('--model_dir', default='model.pkl', help='the pre-trained model')
    opt = parser.parse_args()


    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.model_dir)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir)
        print(testing_accuracy)






