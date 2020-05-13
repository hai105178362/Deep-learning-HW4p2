import numpy as np
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import *
import torchvision
import torch.nn as nn

'''
Loading all the numpy files containing the utterance information and text information
'''
def load_data():
    speech_train = np.load('/home/ubuntu/hw4p2/train_new.npy', allow_pickle=True, encoding='bytes')

    speech_valid = np.load('/home/ubuntu/hw4p2/dev_new.npy', allow_pickle=True, encoding='bytes')

    speech_test = np.load('/home/ubuntu/hw4p2/test_new.npy', allow_pickle=True, encoding='bytes')


    transcript_train = np.load('/home/ubuntu/hw4p2/train_transcripts.npy', allow_pickle=True,encoding='bytes')

    transcript_valid = np.load('/home/ubuntu/hw4p2/dev_transcripts.npy', allow_pickle=True,encoding='bytes')


    return speech_train, speech_valid, speech_test, transcript_train, transcript_valid




'''
Transforms alphabetical input to numerical input, replace each letter by its corresponding 
index from letter_list
'''
def transform_letter_to_index(transcript, letter_list):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    transcript_index = []
    for i in range(len(transcript)):
        new_sentence = []
        new_sentence.append(letter_list.index('<sos>'))
        for j in range(len(transcript[i])-1):
            for k in range(len(transcript[i][j])):
                new_sentence.append(letter_list.index(chr(transcript[i][j][k])))
            new_sentence.append(letter_list.index(' '))
        for m in range(len(transcript[i][-1])):
            new_sentence.append(letter_list.index(chr(transcript[i][-1][m])))
        new_sentence.append(letter_list.index('<eos>'))

        transcript_index.append(new_sentence)
    return transcript_index

LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']




'''
Optional, create dictionaries for letter2index and index2letter transformations
'''
def create_dictionaries(letter_list):
    letter2index = dict()
    index2letter = dict()
    return letter2index, index2letter


class Speech2TextDataset(Dataset):
    '''
    Dataset class for the speech to text data, this may need some tweaking in the
    getitem method as your implementation in the collate function may be different from
    ours. 
    '''
    def __init__(self, speech, text=None, isTrain=True):
        self.speech = speech
        self.isTrain = isTrain
        if (text is not None):
            self.text = text

    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if (self.isTrain == True):
          
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index])
        else:
            return torch.tensor(self.speech[index].astype(np.float32))


def collate_train(batch_data):
    speech,text=zip(*batch_data)
    speech_lens = [len(item) for item in speech]
    text_lens = [len(item) for item in text]
    speech = pad_sequence(speech)
    text = pad_sequence(text)
    return speech,speech_lens,text,text_lens


def collate_test(batch_data):
    speech_lens = [len(item) for item in batch_data]
    speech = pad_sequence(batch_data)
    return speech, speech_lens
