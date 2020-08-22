import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)

        features = features.reshape(features.size(0), -1)

        return features

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
       
        #Producing the Encoder hidden states
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
       
        #Producing the Decoder hidden states
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
       
        #Calculating Alignment scores
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
      
        #Softmaxing the scores
        alpha = self.softmax(att)  # (batch_size, num_pixels)
    
        #Calculating the Context vector
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
    
        return attention_weighted_encoding, alpha


class DecoderRNN(nn.Module):
    def __init__(self, attention_dim, decoder_dim, encoder_dim, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        
        
        self.vocab_size = vocab_size
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.sigmoid = nn.Sigmoid()
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.dropout = 0.5
        self.dropout = nn.Dropout(p=self.dropout)
      

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.lstm_test = nn.LSTMCell(embed_size+encoder_dim+vocab_size, 512, bias=True)
        
        self.linear = nn.Linear(hidden_size*2, vocab_size)
        self.max_seg_length = max_seq_length
    
    #Ankit init_hidden_state START
    def init_hidden_state(self, features):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """

        mean_encoder_out = features.mean(dim=1)
       
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)

        return h,c
          
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""

        #Training and Validation Phase
        if (lengths != 0):  
          
          batch_size  = features.size(0)
          encoder_dim = features.size(-1)
          vocab_size  = self.vocab_size

          features   = features.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
          num_pixels = features.size(1)

          #We embed the input captions
          embeddings = self.embed(captions)
          
          #Initialize the hidden state values
          hiddens, states = self.init_hidden_state(features)  # (batch_size, decoder_dim)
         
          #Calculate the caption lengths for the 5 input captions
          decode_lengths = (lengths[0] - 1)
          decode_lengths = [l-1 for l in lengths]

          #Initialize the predictions tensor and alpha scores tensor
          predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size)
          alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels)

          #Iterate based on the maximum caption length
          for t in range(max(decode_lengths)):
            
            
            batch_size_t = sum([l > t for l in decode_lengths])
        
            #Apply the attention mechanism on the Encoded CNN feature vector and hidden
            #previous hidden state values
            attention_weighted_encoding, alpha = self.attention(features[:batch_size_t],
                                                                  hiddens[:batch_size_t])          


            #New sigmoid gate used for deeper attention mechanism
            gate = self.sigmoid(self.f_beta(hiddens[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)

            #Calculate the new context vector having higher weights for the input image
            attention_weighted_encoding = gate * attention_weighted_encoding

            #Pass the captions embedding along with attention weighted encoding as 
            #input to LSTM
            hiddens, states = self.lstm(
                  torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                  (hiddens[:batch_size_t], states[:batch_size_t]))  # (batch_size_t, decoder_dim)
            
            #Linear layer over the hideen state values would give us the predicted captions
            preds = self.fc(self.dropout(hiddens))

            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

          return predictions, captions, decode_lengths, alphas

        #TESTING PHASE
        if (lengths == 0):

          encoder_out = features
      
          batch_size  = features.size(0)
          encoder_dim = features.size(-1)
          vocab_size  = self.vocab_size

          encoder_out = encoder_out.unsqueeze(1)
          
          num_pixels = encoder_out.size(1)

          #Initialize the hidden state values
          hiddens, states = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

          batch_size_t = 1
          sampled_ids = []
   
          for t in range(self.max_seg_length):
            
            batch_size_t = (encoder_out.size())[1]
            
            #Apply the attention mechanism on the Encoded CNN feature vector and hidden
            #previous hidden state values
            attention_weighted_encoding, alpha = self.attention(encoder_out, hiddens)          
            
            #New sigmoid gate used for deeper attention mechanism
            gate = self.sigmoid(self.f_beta(hiddens))  # gating scalar, (batch_size_t, encoder_dim)
            #Calculate the new context vector having higher weights for the input image
            attention_weighted_encoding = gate * attention_weighted_encoding 

            #For the first iteration since we do not have predictions we initialize 
            #hidden state values and use a special <START> token for this purpose.
            if (t == 0):
  
              hiddens, states = self.lstm(
                    torch.cat([hiddens, attention_weighted_encoding], dim=1),
                    (hiddens, states))  # (batch_size_t, decoder_dim)  

            #After the first iteration we pass the predicted caption as the next input to LSTM  
            if (t != 0):
                      
              hiddens, states = self.lstm_test(
                    torch.cat([inputs, attention_weighted_encoding], dim=1),
                    (hiddens, states))  # (batch_size_t, decoder_dim) 

            #Linear layer over the hideen state values would give us the predicted captions
            preds = self.fc(hiddens.squeeze(1))

            #The predicted caption is passed as the next input to LSTM
            inputs = preds[:]

            #We take the maximum value of the predicted captions         
            _,preds = preds.max(1)
           
            sampled_ids.append(preds)
   
          sampled_ids = torch.stack(sampled_ids, -1)                # sampled_ids: (batch_size, max_seq_length)
          #sampled_ids contains the final predicted caption
          return sampled_ids
          