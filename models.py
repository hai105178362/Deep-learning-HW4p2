import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import *
from dataloader import load_data, collate_train, collate_test, transform_letter_to_index, Speech2TextDataset
import random
from torch.autograd import Variable

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']


class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, lens):
        '''
        :param lens : (1,batch_size)
        :param query :(batch_size, key_size) Query is the output of LSTMCell from Decoder
        :param key: (recording_lens //8,batch_size,key_size) Key Projection from Encoder per time step
        :param value: (recording_lens //8,batch_size,value_size) Value Projection from Encoder per time step
        :return output: Attended Context
        :return attention_mask: Attention mask that can be plotted  
        '''
        key = key.permute(1,0,2)
        attention = (torch.bmm(key, query.unsqueeze(2))).squeeze(2)
        mask = torch.arange(key.size(1)).unsqueeze(0) >= lens.unsqueeze(1)
        mask = mask.to(DEVICE)
        attention.masked_fill_(mask, -1e9)
        attention = nn.functional.softmax(attention, dim=1)

        value = value.permute(1,0,2)
        output = torch.bmm(attention.unsqueeze(1), value).squeeze(1)

        return output, attention







class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    The length of utterance (speech input) can be hundereds to thousands of frames long.
    The Paper reports that a direct LSTM implementation as Encoder resulted in slow convergence,
    and inferior results even after extensive training.
    The major reason is inability of AttendAndSpell operation to extract relevant information
    from a large number of input steps.
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1,batch_first=False,dropout=0.2,bidirectional=True)

    def forward(self, x,lens):
        '''
        :param x :(N, T) input to the pBLSTM
        :return output: (N, T, H) encoded sequence from pyramidal Bi-LSTM 
        '''
        
        recording_length = x.size(0)
        batch_size = x.size(1)
        d_hidden_dim = x.size(2)
        if x.shape[0] % 2 != 0:
            x = x[:-1, :, :]
        x = x.permute(1, 0, 2)
        x = x.reshape(batch_size, recording_length // 2, d_hidden_dim * 2)
        x = x.permute(1, 0, 2)
        lens = lens // 2
        rnn_inp = pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)
        outputs, _ = self.blstm(rnn_inp)
        pblstm_output, pblstm_output_lengths = pad_packed_sequence(outputs)

        return pblstm_output,pblstm_output_lengths



class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key and value.
    Key and value are nothing but simple projections of the output from pBLSTM network.
    '''
    def __init__(self, input_dim, hidden_dim, value_size=128,key_size=128):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=False,dropout=0.2,bidirectional=True)
        self.blstm1 = pBLSTM(hidden_dim*4,hidden_dim)
        self.blstm2 = pBLSTM(hidden_dim*4,hidden_dim)
        self.blstm3 = pBLSTM(hidden_dim*4,hidden_dim)
        self.key_network = nn.Linear(hidden_dim*2, value_size)
        self.value_network = nn.Linear(hidden_dim*2, key_size)
        self.lockeddropout = LockedDropout()

    def forward(self, x, lens):
        x = x.to(DEVICE)

        rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)
        outputs, _ = self.lstm(rnn_inp)
        encoded_input, encoded_input_lengths = utils.rnn.pad_packed_sequence(outputs)
        encoded_input = self.lockeddropout(encoded_input)
        pblstm1_outputs,pblstm1_outputs_length = self.blstm1(encoded_input,encoded_input_lengths)
        pblstm1_outputs = self.lockeddropout(pblstm1_outputs)

        pblstm2_outputs,pblstm2_outputs_length = self.blstm2(pblstm1_outputs,pblstm1_outputs_length)
        pblstm2_outputs = self.lockeddropout(pblstm2_outputs)

        pblstm3_outputs,pblstm3_outputs_length = self.blstm3(pblstm2_outputs,pblstm2_outputs_length)
        keys = self.key_network(pblstm3_outputs)
        value = self.value_network(pblstm3_outputs)

        return keys, value, pblstm3_outputs_length

class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step, 
    thus we use LSTMCell instead of LSLTM here.
    The output from the second LSTMCell can be used as query here for attention module.
    In place of value that we get from the attention, this can be replace by context we get from the attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
    def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=True):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=hidden_dim + value_size, hidden_size=hidden_dim)
        self.lockeddropout = LockedDropout()
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)

        self.isAttended = isAttended
        if (isAttended == True):
            self.attention = Attention()

        self.character_prob = nn.Linear(key_size + value_size, vocab_size)
        self.teacher_forcing_ratio = 1
        self.value_size = value_size

    def forward(self, key, values, lens, text=None, isTrain=True):
        '''
        :param key :(seq_length, batch_size, key_size) Output of the Encoder Key projection layer
        :param values: (seq_length, batch_size, value_size) Output of the Encoder Value projection layer
        :param text: (text_len,batch_size) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :param lens: time_step
        :return predictions: Returns the character perdiction probability 
        '''
        
        batch_size = key.shape[1]

        if (isTrain):
            max_len =  text.shape[0]
            embeddings = self.embedding(text)
        else:
            max_len = 250

        predictions = []
        hidden_states = [None, None]
        prediction = torch.zeros(batch_size,1).to(DEVICE)
        context = torch.zeros(batch_size,self.value_size)
        for i in range(max_len-1):
            
            if (isTrain):
                if random.random() > self.teacher_forcing_ratio:

                    char_embed = self.embedding(prediction.argmax(dim=-1))

                else:
                    if i == 0:
                        char_embed = self.embedding(prediction.argmax(dim=-1))

                    else:
                        char_embed = embeddings[i,:,:]
            else:
                char_embed = self.embedding(prediction.argmax(dim=-1))
               
            char_embed = char_embed.to(DEVICE)
            context = context.to(DEVICE)
            inp = torch.cat([char_embed, context], dim=1)
          
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            inp_2 = inp_2.unsqueeze(0)
            inp_2 = self.lockeddropout(inp_2)
            inp_2 = inp_2.squeeze(0)
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            output = hidden_states[1][0]
          
            context,_ = self.attention(output, key, values, lens)

            prediction = self.character_prob(torch.cat([output, context], dim=1))
            predictions.append(prediction.unsqueeze(1))
          

        predictions = torch.cat(predictions, dim=1)
       

        return predictions


class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=True):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, hidden_dim)

    def forward(self, speech_input, speech_len, text_input=None, isTrain=True):
        key, value,lens = self.encoder(speech_input, speech_len)
        if (isTrain == True):
            predictions = self.decoder(key, value, lens, text_input)
        else:
            predictions = self.decoder(key, value, lens, text=None, isTrain=False)
        return predictions

class LockedDropout(nn.Module):
    '''
    Args:
    p (float): Probability of an element in the dropout mask to be zeroed.
    '''

    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [sequence length, batch size, rnn hidden size]): Input to
            apply dropout too.
        """
        if not self.training or not self.p:
            return x
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask


