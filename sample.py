import os
import time
import pickle
import json
import torch
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
import dataload
import Vocab_builder
import model

#from Vocab_builder import Vocab_builder
from dataload import DATALOAD
from model import EncoderCNN
from model import DecoderRNN
from os import listdir
from os.path import join, isdir
import timeit
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    
    test_dir = '/content/drive/My Drive/Processed Data/test'  
    test_data_url=[d for d in listdir(test_dir+"/images") if d.lower().endswith(('.png', '.jpg', '.jpeg'))]


    x=open("/content/drive/My Drive/Processed Data/train/captions.txt","r")
    train_captions_text=x.readlines()
    x=open("/content/drive/My Drive/Processed Data/test/captions.txt","r")
    test_captions_text=x.readlines()

    caption_dict = dataload.loadcaptions(train_captions_text)
    vocab = Vocab_builder.Vocab_builder(caption_dict = caption_dict, threshold = 3)   

 
    flickr_mean_color = [0.44005906636983594, 0.4179391194853607, 0.3848489007106294]
    flickr_std_color = [0.28628332805186396, 0.2804168453671926, 0.29043924853401465]

    transforms = transforms.Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(flickr_mean_color,flickr_std_color),])
    
    
    vocab_size = vocab.index
    print("vocab size",vocab_size)
    hidden_dim = 512
    embedding_dim = 256

    # Initializing the Encoder and Decoder Network with arguments passed
    encoder2 = EncoderCNN(embed_size=embedding_dim).eval()
    decoder2 = DecoderRNN(embed_size=embedding_dim, hidden_size=hidden_dim, vocab_size = vocab_size,num_layers=1,max_seq_length=20)
 
    # Path where the input saved module is present    
    decoder_saved_module = '/content/drive/My Drive/Processed Data/iter3_1decoder.pt'
    encoder_saved_module = '/content/drive/My Drive/Processed Data/iter3_1encoder.pt'

 
    encoder2.load_state_dict(torch.load(encoder_saved_module))
    decoder2.load_state_dict(torch.load(decoder_saved_module))

    # Taking input from user for image to be captioned
    img = "3213992947_3f3f967a9f.jpg"
    image_path = os.path.join(test_dir+"/images/", img)
    print(image_path)
    
    captions=dataload.loadcaptions(test_captions_text)
    actual_caption=[]
    for a in range(len(captions)):
      for key,value in captions[a].items():
        if key==img:
          actual_caption=value
    print(actual_caption)                   
    image = Image.open(image_path)
    plt.imshow(np.asarray(image))

    image = transforms(Image.open(image_path))
    image = image.unsqueeze(0)
    
    if torch.cuda.is_available():
        encoder2.cuda()
        decoder2.cuda()
        image = Variable(image).cuda()
    else:
        image = Variable(image)
    print(image.shape)

    # Passing the input from the network 
    encoder_out = encoder2(image)
    decoder_out = decoder2.sample(encoder_out)

    # Printing the outputs
    predicted1 = vocab.get_sentence(decoder_out[0])
    print("predicted caption:",predicted1)
    