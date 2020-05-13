import time
import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import *


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']




def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    model.to(DEVICE)
    start = time.time()

    perplexity = 1
    for batch_num, items in enumerate(train_loader):

        optimizer.zero_grad()
        speech, speech_lens, text, text_lens = items
        speech, text = speech.to(DEVICE), text.to(DEVICE)
        text_lens = torch.LongTensor(text_lens).to(DEVICE)

        predictions = model(speech, speech_lens, text, isTrain=True)

        text_T = text.permute(1,0)
        new_l = text_T.size(1) - 1
        mask = torch.zeros((text_T.size(0), new_l))
        for i in range(len(text_lens)):
            mask[i, :text_lens[i]-1] = 1
        mask = mask.view(-1).to(DEVICE)
        inputs = predictions.view(-1,predictions.size(2))
        targets = text_T[:,1:].reshape(-1)
        loss = criterion(inputs,targets)
        masked_loss = loss * mask

        masked_loss = torch.sum(masked_loss)
        masked_loss.backward()
        perplexity = perplexity * torch.exp(torch.sum(masked_loss) / torch.sum(mask))
        torch.nn.utils.clip_grad_norm_(model.parameters(),2)
            
        optimizer.step()

        
        if batch_num % 50 == 49:
            print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f} Perplexity: {}'.format(epoch + 1, batch_num + 1, masked_loss,perplexity))
            PATH = './new1.pth'
            torch.save(model.state_dict(), PATH)

        torch.cuda.empty_cache()




def test(model, test_loader):
    model.eval()
    model.to(DEVICE)
    result_list = []
    for batch_num, items in enumerate(test_loader):
        
        speech, speech_lens = items
        speech = speech.to(DEVICE)
        speech_lens = torch.LongTensor(speech_lens).to(DEVICE)

        predictions = model(speech, speech_lens,isTrain=False)
        out_list = []
        for x in predictions:
            predict_transcipt = ''
            for w in x:
                char = LETTER_LIST[w.argmax()]
                if char != '<eos>':
                    predict_transcipt += char
                else:
                    break
            out_list.append(predict_transcipt)
        result_list += out_list
 

    return result_list


