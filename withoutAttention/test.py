## For pretty print of tensors.
## Must be located at the first line except the comments.
from __future__ import print_function

## Import the basic modules.
import argparse
import numpy as np
from os import listdir
from os.path import join, isdir
import timeit
import matplotlib.pyplot as plt
## Import the PyTorch modules.
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import dataload
import Vocab_builder
import dataload
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
## You are supposed to implement the following four source codes:
## {softmax.py, twolayernn.py, convnet.py, mymodel.py

import model 
from model import EncoderCNN
from model import DecoderRNN
#import EncoderCNN, DecoderRNN

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
                    choices=['Pretrained','test'],
                    help='which model to train/evaluate')
parser.add_argument('--hidden-dim', type=int, help='number of hidden features/activations')
parser.add_argument('--embed-dim', type=int, help='embed layer dimensions')
parser.add_argument('--kernel-size', type=int, help='size of convolution kernels/filters')
parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adagrad', 'adadelta'], help='which optimizer')
parser.add_argument('--checkpoint', type=str, help='checkpoint for loading saved model dictionary')
## Add more command-line options for other configurations.
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
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


if __name__ == '__main__':
    root="/content/drive/My Drive/Colab Notebooks/576_project/Processed Data/"
    test_dir =root+"test"
    test_data_url = [ d for d in listdir(test_dir+"/images") if d.lower().endswith(('.png', '.jpg', '.jpeg'))]
    x=open(test_dir+"/captions.txt","r")
    test_captions_text=x.readlines()
    print('STARTING THE TESTING PHASE ...........')

    x=open(root+"/train/captions.txt","r")
    train_captions_text=x.readlines()
    # Reading the vocab file
    caption_dict = dataload.loadcaptions(train_captions_text)
    vocab = Vocab_builder.Vocab_builder(caption_dict = caption_dict, threshold = 3)   

    # Transforming the image file by Resizing, making tensor from it and then 
    # Normalizing the image by mean and standard deviation
    flickr_mean_color = [0.44005906636983594, 0.4179391194853607, 0.3848489007106294]
    flickr_std_color = [0.28628332805186396, 0.2804168453671926, 0.29043924853401465]

    transform = transforms.Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(flickr_mean_color,flickr_std_color),])
    
    vocab_size = vocab.index
    print("vocab size",vocab_size)

    test_dataset = dataload.DATALOAD(test_dir,test_data_url,test_captions_text[:50],vocab, split='test', download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=dataload.val_collate, **kwargs)

   
    if args.model == 'test':
    # Initializing the Encoder and Decoder Network with arguments passed
      encoder2 = EncoderCNN(embed_size=args.embed_dim).eval()
      decoder2 = DecoderRNN(embed_size=args.embed_dim, hidden_size=args.hidden_dim, vocab_size = vocab_size,num_layers=1,max_seq_length=20)
    else:
      raise Exception('Unknown model {}'.format(args.model))
    # Path where the input saved module is present    
    decoder_saved_module = '/content/drive/My Drive/Colab Notebooks/576_project/version5/lr0.01_2decoder.pt'
    encoder_saved_module = '/content/drive/My Drive/Colab Notebooks/576_project/version5/lr0.01_2encoder.pt'
    
   
    ## Activate CUDA if specified and available.
    if args.cuda:
       encoder2.cuda()
       decoder2.cuda()
    # Reading the pretrained weights for Encoder and Decoder

    encoder2.load_state_dict(torch.load(encoder_saved_module))
    
    decoder2.load_state_dict(torch.load(decoder_saved_module))

    for batch_idx, batch in enumerate(test_loader):
        image,caption,urls = batch[0],batch[1],batch[2]
       
        if args.cuda:
            image,caption = image.cuda(), caption.cuda()

            # Read images and their target labels in the current batch.
            image,caption = Variable(image),caption
    

        ground_truth = []
        predicted = []
      
    # Passing the input from the network 
        encoder_out = encoder2(image)
  
        decoder_out = decoder2.sample(encoder_out)
        print(decoder_out.size())
        
        #fetching the generated word sequence from the vocabulary for the predicted captions
        for i in range(len(decoder_out)):
          sampled_seq =vocab.get_sentence(decoder_out[i])
          predicted.append(sampled_seq)
        
        #storing the ground truth of each image
        #removing end token from the sequence
        for i in range(len(caption)):

          targets = [c[0:-1] for c in caption[i]]
          ground_truth.append(targets)

        #printing the predicted and ground truth for each image in the batch
        for i in range(len(predicted)):
           print("prediction",predicted[i])
           print("ground truth",ground_truth[i])
       
        print("batch_idx and their bleu scores", batch_idx)
        print("bleu 4",corpus_bleu(ground_truth, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
        print("bleu 3",corpus_bleu(ground_truth, predicted, weights=(1/3, 1/3, 1/3, 0)))
        print("bleu 2",corpus_bleu(ground_truth, predicted, weights=(0.5, 0.5, 0, 0)))
        print("bleu 1",corpus_bleu(ground_truth, predicted, weights=(1, 0, 0, 0)))


    """To print images
    fig = plt.figure(figsize=(1,1))
    fig.add_subplot(1,1,1).imshow((image[0].reshape(224,224,3)).numpy().astype('uint8'))
    fig.savefig('test1.png')
    """

    print("Test completed")
   
