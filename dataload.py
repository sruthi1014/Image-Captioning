from __future__ import print_function
import torch
import os
from os import listdir
import os.path
import errno
import numpy as np
import random
import sys
import nltk
import torch.utils.data as data
from PIL import Image
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class DATALOAD():
    def __init__(self,root,url, captions,vocab, split='train',
                 transform=None, download=False):
        self.root = root
        self.trans = transform
        self.split = split  # train, val, or test
        self.url= url
        self.data=url
        self.captions=captions
        self.vocab=vocab

        if download:
            self.download()
        self.transform()

        if self.split == 'train': 
           self.train_data=self.data
           self.train_captions=self.captions
           self.train_url=self.url

        elif self.split == 'val':
           self.val_data=self.data
           self.val_captions=self.captions
           self.val_url=self.url

        elif self.split == 'test':
           self.test_data=self.data
           self.test_captions=self.captions
           self.test_url=self.url
    
    def __len__(self):
      
        if self.split == 'train': 
           return len(self.train_data)
        elif self.split == 'val':
           return len(self.val_data)
        elif self.split == 'test':
           return len(self.test_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
       
        if self.split == 'train':
            img,caption,url = self.train_data[index], self.train_captions[index],self.train_url[index]
            
        elif self.split == 'val':
           img,caption,url = self.val_data[index], self.val_captions[index],self.val_url[index]
  
        elif self.split == 'test':
           img, caption,url = self.test_data[index], self.test_captions[index],self.test_url[index]
        
        if self.trans is not None:
            img = self.trans(img)
        #specific condition as train and value has each image and 1 annotation where as test has one image with list of 5 annotations as each element
        if  self.split !='test':  
            caption=torch.Tensor(caption)
            
        return img,caption,url
    
    def download(self):
        # importing and extracting flickr images
        import tarfile
        tf = tarfile.open("/content/drive/My Drive/Colab Notebooks/576_project/shannon.cs.illinois.edu/DenotationGraph/data/flickr8k-images.tar")
        tf.extractall(path=".")
       # importing the tokenized captions
        tf2 = tarfile.open("/content/drive/My Drive/Colab Notebooks/576_project/data/flickr8k.tar.gz")
        tf2.extractall(path=".")
        
    def transform(self):
        i=0
        dataset=[]
        captions=[]
       
       #converting the string format to list
        captions=loadcaptions(self.captions)
        
        cap=[] 
        urls=[]
        print( ' we are here check 1')
  
        #structuring the captions and getting image path and caption for each element in the caption list
        for a in range(len(captions)):
                
                for key,value in captions[a].items():
                    cap.append(value)
                    #print(cap)
                    urls.append(self.root+"/images/"+key)
                           
        dataset=urls
        
        print("check 2")   
        self.url=urls
      
        y=[]
       	images=[]
        target=[]
        new_url=[]
        num=5
        #for train and validation we extend the list to map each annotation for an image. eg: imageid-cap1, imageid1-cap2, imageid1-cap3,imageid1-cap4,imageid1-cap5
        for i in range(len(cap)):
              if self.split !='test':
                 images.extend([dataset[i]]*num)
                 new_url.extend([self.url[i]]*num)
                 for k in range(num):
                         y.append(cap[i][k])
            
        #tokenizing the caption for train and validation inorder to check loss later with prediction
        if self.split !='test':
           self.url=new_url
           for i in range(len(y)):            
              token = nltk.tokenize.word_tokenize(y[i].lower())
        #print(token)
              vec = []
              vocab=self.vocab
              vec.append(vocab.get_id('<start>'))
              vec.extend([vocab.get_id(word) for word in token])
              vec.append(vocab.get_id('<end>'))   
              target.append(vec)
        else:
              target=cap
              images=dataset
        #opening the images from their respective paths
        for i in range(len(images)):
            images[i]=Image.open(images[i])

        self.data=images  
        self.captions=target
      
        return images,target

def collate_fn(data):
    # Sort a data list by caption length (descending order).
      
      data.sort(key=lambda x: len(x[1]), reverse=True)
      images, captions,url = zip(*data)
   
    # Merge images (from tuple of 3D tensor to 4D tensor).
      images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
      lengths = [len(cap) for cap in captions]
      #print(lengths)

      #padding with the max length within the batch
      targets = torch.zeros(len(captions), max(lengths)).long()
      for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]         
      return images, targets, lengths

def val_collate(data):
    images,captions,url=zip(*data)
    images = torch.stack(images, 0)
    return images,captions,url

def loadcaptions(text):
        captions=[]
        import json 
       
        for i in  text:
          res = json.loads(i) 
          captions.append(res)
          
        return captions
        
 
    
        