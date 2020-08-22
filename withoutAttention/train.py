## For pretty print of tensors.
## Must be located at the first line except the comments.
from __future__ import print_function

## Import the basic modules.
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
## Import the PyTorch modules.
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import dataload
import Vocab_builder

import model 
from model import EncoderCNN
from model import DecoderRNN


## Initilize a command-line option parser.
parser = argparse.ArgumentParser(description='Flickr8k')

## Add a list of command-line options that users can specify.
## Shall scripts (.sh) files for specifying the proper options are provided.
parser.add_argument('--lr', type=float, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, metavar='M', help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N', help='input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N', help='number of epochs to train')
parser.add_argument('--model',
                    choices=['Pretrained'],
                    help='which model to train/evaluate')
parser.add_argument('--hidden-dim', type=int, help='number of hidden features/activations')
parser.add_argument('--embed-dim', type=int, help='embed layer dimensions')
parser.add_argument('--kernel-size', type=int, help='size of convolution kernels/filters')
parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adagrad', 'adadelta'], help='which optimizer')

## Add more command-line options for other configurations.
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='number of batches between logging train status')

## Parse the command-line option.
args = parser.parse_args()

## CUDA will be supported only when user wants and the machine has GPU devices.
args.cuda = not args.no_cuda and torch.cuda.is_available()

## Change the random seed.
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

## Set the device-specific arguments if CUDA is available.
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

im_size = (3, 224, 224)
n_classes = 10


from os import listdir
from os.path import join, isdir
import timeit


# number of channels of the dataset image, 3 for color jpg

#root path where train, val, test folders are available
root="/content/drive/My Drive/Colab Notebooks/576_project/Processed Data/"
CHANNEL_NUM = 3
train_root = root+"train" #train folder path
val_root=root+"dev"    #val folder path

#the below 3 lines fetches the list of all images( png,jpeg,jpg) in their respective path and stores the image id 
train_data_url = [d for d in listdir(train_root+"/images") if d.lower().endswith(('.png', '.jpg', '.jpeg'))]
val_data_url=[d for d in listdir(val_root+"/images") if d.lower().endswith(('.png', '.jpg', '.jpeg'))]


#this fetches the captions text file for each train, val  split
x=open(train_root+"/captions.txt","r")
train_captions_text=x.readlines()
x=open(val_root+"/captions.txt","r")
val_captions_text=x.readlines()


## Normalize each image by subtracdting the mean color and divde by standard deviation.
## For convenience, per channel mean color and standard deviation are provided.

flickr_mean_color = [0.44005906636983594, 0.4179391194853607, 0.3848489007106294]
flickr_std_color = [0.28628332805186396, 0.2804168453671926, 0.29043924853401465]
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(flickr_mean_color, flickr_std_color),
])

#building vocabulary by considering the train caption list
print('Building Vocabulary')
caption_dict = dataload.loadcaptions(train_captions_text)
vocab = Vocab_builder.Vocab_builder(caption_dict = caption_dict, threshold = 3)   
vocab_size = vocab.index
print(vocab.index)


## Load training, validation
## Apply the normalizing transform uniformly across train and validation datasets.
train_dataset = dataload.DATALOAD(train_root,train_data_url,train_captions_text,vocab, split='train', download=False, transform=transform)
#print(train_dataset)
val_dataset = dataload.DATALOAD(val_root,val_data_url,val_captions_text,vocab, split='val', download=False, transform=transform)


print("hey check 3")
## DataLoaders provide various ways to get batches of examples.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=dataload.collate_fn, **kwargs)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=dataload.collate_fn, **kwargs)

## Load the proper neural network model.
if args.model == 'Pretrained':
   
    model.encoder = model.EncoderCNN(args.embed_dim)
    model.decoder=model.DecoderRNN(embed_size = args.embed_dim, hidden_size = args.hidden_dim, vocab_size = vocab_size,num_layers=1,max_seq_length=10)

else:
    raise Exception('Unknown model {}'.format(args.model))

## the loss function -cross-entropy.

criterion = functional.cross_entropy

## Activate CUDA if specified and available.
if args.cuda:
    model.encoder.cuda()
    model.decoder.cuda()

params = list(model.encoder.linear.parameters()) + list(model.encoder.bn.parameters())+ list(model.decoder.parameters())

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters(), args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(params, args.lr, betas=(0.9,args.momentum), weight_decay=args.weight_decay)
elif args.optimizer == 'adadelta':
    optimizer = torch.optim.Adadelta(model.parameters(), args.lr, rho=0.9, weight_decay=args.weight_decay)
pass


## Function: train the model for each epoch
def train(epoch):
    #  model is a class that inherits nn.Module 
    # This puts the model in train mode as opposed to eval mode, so it knows which one to use.
    print("check 5")
    model.encoder.train()
   
    #print(" check lalala")
    model.decoder.train()
    print("check 6")
    # print(model.fc)
    # For each batch of training images,
    cum_train_loss=0
    cum_val_loss=0
    for batch_idx, batch in enumerate(train_loader):
        # Read images and their target labels in the current batch.
        
        images,captions,lengths = Variable(batch[0]),Variable(batch[1]),batch[2]

        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
   
        # Load the current training example in the CUDA core if available.
        if args.cuda:
            images= images.cuda()

    
        features = model.encoder(images)
        output=model.decoder(features,captions,lengths)
        
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, targets)
       
        model.decoder.zero_grad()
        model.encoder.zero_grad()
        loss.backward()
        optimizer.step()

        pass
        cum_train_loss +=loss
        
        # Print out the loss and accuracy on the first 10 batches of the validation set.
        #  adjusting the printing frequency by changing --log-interval option in the command-line.
        if batch_idx % args.log_interval == 0:
            # Compute the average validation loss and accuracy.
            val_loss=evaluate('val', n_batches=10)
            # Compute the training loss.
            train_loss = loss.data.item()
  
            # Compute the number of examples in this batch.
            examples_this_epoch = batch_idx * len(images)

            # Compute the progress rate in terms of the batch.
            epoch_progress = 100. * batch_idx / len(train_loader)

            # Print out the training loss, validation loss, and accuracy with epoch information.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Train Loss: {:.6f}\tVal Loss:{:.6f}\t'.format(
                epoch, examples_this_epoch, len(train_loader.dataset),
                epoch_progress, train_loss, val_loss))
        cum_val_loss+=val_loss
    avg_val_loss=cum_val_loss/(batch_idx+1)
    avg_train_loss=cum_train_loss/(batch_idx+1)
    print('Train Epoch: {}\t'
           'Avg Train Loss: {:.6f}\t Val Loss:{:.6f}\t'.format(epoch,avg_train_loss,avg_val_loss))

## Function: evaluate the learned model on either validation or test data.
def evaluate(split, verbose=False, n_batches=None):
    # Recall model is a class that inherits nn.Module that we learned in the class.
    # This puts the model in eval mode as opposed to train mode, so it knows which one to use.
    model.encoder.eval()
    model.decoder.eval()
    # Initialize cumulative loss and the number of correctly predicted examples.
    loss = 0
    correct = 0
    n_examples = 0

    # Load the correct dataset between validation.
    if split == 'val':
        loader = val_loader
    

    # For each batch in the loaded dataset,
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
          
            data,caption,lengths = batch[0],batch[1],batch[2]
          
            targets = pack_padded_sequence(caption, lengths, batch_first=True)[0]
            
            # Load the current training example in the CUDA core if available.
            if args.cuda:
                data,caption = data.cuda(), caption.cuda()

            # Read images and their target labels in the current batch.
            data,caption = Variable(data),Variable(caption)

            # Measure the output results given the data.
            features = model.encoder(data)
            output=model.decoder(features,caption,lengths)
            
            # Accumulate the loss by comparing the predicted output and the true targets ( both are in pack padded sequence).
            loss += criterion(output, targets).data
          
         
            # Skip the rest of evaluation if the number of batches exceed the n_batches.
            if n_batches and (batch_i >= n_batches):
                break

    # Compute the average loss per example.
    loss /= (batch_i+1)

   
    # If verbose is True, then print out the average loss and accuracy.
    if verbose:
        print('\n{} set: Average loss: {:.4f}'.format(
            split, loss))
    return loss


## Train the model one epoch at a time.
for epoch in range(1, args.epochs + 1):
    print("check 4")
    train(epoch)
    #save the model for every epoch
    save_every=1
    if epoch % save_every == 0:
            torch.save(model.encoder.state_dict(), 'lr0.005_%dencoder.pt'%(epoch))
            torch.save(model.decoder.state_dict(), 'lr0.005_%ddecoder.pt'%(epoch))

print("Training completed")



"""
# Later you can call torch.load(file) to re-load the trained model into python
# See http://pytorch.org/docs/master/notes/serialization.html for more details
"""