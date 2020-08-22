## For pretty print of tensors.
## Must be located at the first line except the comments.
from __future__ import print_function

## Import the basic modules.
import argparse
import numpy as np
import time
## Import the PyTorch modules.
import model
import torch
import torch.nn as nn
import dataload
import Vocab_builder
import torch.nn.functional as functional
import torch.optim as optim
from   torchvision import transforms
from   torch.autograd import Variable
from   torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
print_freq = 1 #test parameters
epochs = 120  #test parameters
grad_clip = 5.  # clip gradients at an absolute value
global split


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
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='number of batches between logging train status')
#parser.add_argument('--flickr8k', default='data',
                   # help='directory that contains cifar-10-batches-py/ (downloaded automatically if necessary)')

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

root="/content/drive/My Drive/IDS 576 Milestone/Processed Data/"
CHANNEL_NUM = 3
train_root = root+"train" #train folder path
val_root=root+"dev"    #val folder path

train_root = root+"train" #train folder path
val_root=root+"dev"    #val folder path
test_root=root+"test"  #test folder path

train_data_url = [d for d in listdir(train_root+"/images")]
val_data_url=[d for d in listdir(val_root+"/images")]
test_data_url = [ d for d in listdir(test_root+"/images")]

#this fetches the captions text file for each train, val  split
x=open(train_root+"/captions.txt","r")
train_captions_text=x.readlines()
x=open(val_root+"/captions.txt","r")
val_captions_text=x.readlines()
x=open(test_root+"/captions.txt","r")
test_captions_text=x.readlines()


## Normalize each image by subtracdting the mean color and divde by standard deviation.
## For convenience, per channel mean color and standard deviation are provided.

flickr_mean_color = [0.44005906636983594, 0.4179391194853607, 0.3848489007106294]
flickr_std_color = [0.28628332805186396, 0.2804168453671926, 0.29043924853401465]
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(flickr_mean_color, flickr_std_color),
])


global vocab
caption_dict = dataload.loadcaptions(train_captions_text)
vocab = Vocab_builder.Vocab_builder(caption_dict = caption_dict, threshold = 3)   
global vocab_size
vocab_size = vocab.index

## Load training, validation, and test data separately.
## Apply the normalizing transform uniformly across three dataset.
train_dataset = dataload.DATALOAD(train_root,train_data_url,train_captions_text,vocab, split='train', download=False, transform=transform)
val_dataset = dataload.DATALOAD(val_root,val_data_url,val_captions_text,vocab, split='val', download=False, transform=transform)
test_dataset = dataload.DATALOAD(test_root,test_data_url,test_captions_text,vocab, split='test', download=False, transform=transform)

#attention_dim, decoder_dim, encoder_dim

attention_dim = 512  #Dimension of attention linear layers
decoder_dim   = 512  #Dimension of decoder RNN
encoder_dim   = 2048 #Dimension of encoder RNN
emb_dim       = 512  # dimension of word embeddings


global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

## DataLoaders provide various ways to get batches of examples.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=dataload.collate_fn, **kwargs)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=dataload.collate_fn, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=dataload.val_collate, **kwargs)
## Load the proper neural network model.
if args.model == 'Pretrained':
    # Problem 2 (no hidden layer, input -> output)
    model.encoder = model.EncoderCNN(10)
    model.decoder = model.DecoderRNN(encoder_dim = 2048, decoder_dim=512, attention_dim=512, embed_size = 512, hidden_size = args.hidden_dim, vocab_size = vocab_size,num_layers=1,max_seq_length=15)
#elif args.model == 'resnet_common':
    # Problem 5 (multiple hidden layers, input -> hidden layers -> output)
 #   print("sruthi check 1")
  #  model = models.resnetcommon.ResnetCommon(im_size, args.hidden_dim, args.kernel_size, n_classes)

else:
    raise Exception('Unknown model {}'.format(args.model))

## Deinfe the loss function as cross-entropy.
## This is the softmax loss function (i.e., multiclass classification).
criterion = functional.cross_entropy

## Activate CUDA if specified and available.
if args.cuda:    
    model.cuda()
    
params = list(model.encoder.linear.parameters()) + list(model.decoder.parameters())

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters(), args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'adam':
    #optimizer = torch.optim.Adam(params, args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(params, args.lr, betas=(0.9,args.momentum), weight_decay=args.weight_decay)
elif args.optimizer == 'adadelta':
    optimizer = torch.optim.Adadelta(model.parameters(), args.lr, rho=0.9, weight_decay=args.weight_decay)
# optimizer = torch.optim.Adam(model.parameters(),lr = args.lr,weight_decay = args.weight-decay)
pass


#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


## Function: train the model just one iteration.
def train(epoch):
    # Recall model is a class that inherits nn.Module that we learned in the class.
    # This puts the model in train mode as opposed to eval mode, so it knows which one to use.
 
    
    model.encoder.train()
    model.decoder.train()

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.decoder.parameters()),
                                             lr=decoder_lr)
 
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.encoder.parameters()),
                                             lr=encoder_lr)
    
    # For each batch of training images,
    cum_train_loss=0
    cum_val_loss=0

    # For each batch of training images,
    for batch_idx, batch in enumerate(train_loader):
          
        images,captions,lengths = Variable(batch[0]),batch[1],batch[2]
  
        captions_new=captions

        captions_new = captions_new[:, 1:]

        targets = pack_padded_sequence(captions_new, lengths, batch_first=True)[0]

        targets = targets[targets.nonzero()]
        targets = targets.squeeze(1)
  
        # Load the current training example in the CUDA core if available.
        if args.cuda:
            images= images.cuda()

        features = model.encoder(images)
        
        output, caps_sorted, decode_lengths, alphas  = model.decoder(features,captions,lengths)
        
        output = pack_padded_sequence(output, decode_lengths, batch_first=True)[0]

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, targets) 
  

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

      
        model.decoder.zero_grad()
        model.encoder.zero_grad()
        
        loss.backward()
        optimizer.step()
        
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        cum_train_loss +=loss
        
        # Print out the loss and accuracy on the first 4 batches of the validation set.
        # You can adjust the printing frequency by changing --log-interval option in the command-line.
        if batch_idx % args.log_interval == 0:
            # Compute the average validation loss and accuracy.
            val_loss=evaluate('val', n_batches=10)
            #print("check 11")
            # Compute the training loss.
            train_loss = loss.data.item()
            #print(train_loss)
            # Compute the number of examples in this batch.
            examples_this_epoch = batch_idx * len(images)

            # Compute the progress rate in terms of the batch.
            epoch_progress = 100. * batch_idx / len(train_loader)
            #print(epoch_progress)
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
       
    # Load the correct dataset between validation and test data based on the split option.
    if split == 'val':
        loader = val_loader

    elif split == 'test':
        loader = test_loader
    
    if (split == 'val'):
      # For each batch in the loaded dataset,
      with torch.no_grad():
          for batch_i, batch in enumerate(loader):

              data,caption,lengths = Variable(batch[0]),batch[1],batch[2]

              caption = caption[:, 1:]

              targets = pack_padded_sequence(caption, lengths, batch_first=True)[0]
              targets = targets[targets.nonzero()]
              targets = targets.squeeze(1)          
              
              # Load the current training example in the CUDA core if available.
              if args.cuda:
                  data,caption = data.cuda(), caption.cuda()

              # Measure the output results given the data.
              features = model.encoder(data)
              
              output, caps_sorted, decode_lengths, alphas = model.decoder(features,caption,lengths)
              
              # Accumulate the loss by comparing the predicted output and the true target labels.
              output = pack_padded_sequence(output, decode_lengths, batch_first=True)[0]
              
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

    if (split == 'test'):
      
      # For each batch in the loaded dataset,
      with torch.no_grad():
          for batch_i, batch in enumerate(loader):
                      
              data,caption,lengths = batch[0],batch[1],batch[2]
                   
              caption_or = caption[:]
          
              # Load the current training example in the CUDA core if available.
              if args.cuda:
                  data,caption = data.cuda(), caption.cuda()
      
              # Measure the output results given the data.
              features = model.encoder(data)

              #Start of Ankit
              caption = 0
              lengths = 0
              
              output=model.decoder(features,caption,lengths)
              output = output.squeeze(1)
  
              ground_truth = []
              predicted = []

              for i in range(len(output)):
                sampled_seq =vocab.get_sentence(output[i])
                predicted.append(sampled_seq)


              for i in range(len(caption_or)):

                targets = [c[0:-1] for c in caption_or[i]]

                ground_truth.append(targets)

              print("bleu 4",corpus_bleu(ground_truth, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
              print("bleu 1",corpus_bleu(ground_truth, predicted, weights=(1, 0, 0, 0)))
         

## Train the model one epoch at a time.
for epoch in range(1, args.epochs + 1):
    print("check 4")
    train(epoch)
    save_every=1
    if epoch % save_every == 0:
            torch.save(model.encoder.state_dict(), 'new_%dencoder.pt'%(epoch))
            torch.save(model.decoder.state_dict(), 'new_%ddecoder.pt'%(epoch))

## Evaluate the model on the test data and print out the average loss and accuracy.
## Note that you should use every batch for evaluating on test data rather than just the first four batches.
evaluate('test', verbose=True)

## Save the model (architecture and weights)
#torch.save(model, args.model + '.pt')
print("COMPLETED")

"""
# Later you can call torch.load(file) to re-load the trained model into python
# See http://pytorch.org/docs/master/notes/serialization.html for more details
"""