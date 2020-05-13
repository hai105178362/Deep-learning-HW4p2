import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import *
from models import Seq2Seq
from train_test import train, test
from dataloader import load_data, collate_train, collate_test, transform_letter_to_index, Speech2TextDataset
import csv
from torch.optim.lr_scheduler import StepLR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

def save_to_csv(data_list):
    with open("/home/ubuntu/hw4p2/hw4_p2_submission.csv", "w") as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(['Id','Predicted'])
        for i in range(len(data_list)):
            writer.writerow([i,data_list[i]])

def main():
    model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), hidden_dim=128)
    learningRate = 0.001
    weightDecay = 5e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,weight_decay=weightDecay)
    criterion = nn.CrossEntropyLoss(reduction='none')
    nepochs = 40
    batch_size = 64 if DEVICE == 'cuda' else 1

    speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()
    character_text_train = transform_letter_to_index(transcript_train, LETTER_LIST)
    character_text_valid = transform_letter_to_index(transcript_valid, LETTER_LIST)

    train_dataset = Speech2TextDataset(speech_train, character_text_train)
    val_dataset = Speech2TextDataset(speech_valid, character_text_valid)
    test_dataset = Speech2TextDataset(speech_test, None, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_test)

    model.train()
    model.load_state_dict(torch.load('./new1.pth'))
    model.to(DEVICE)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    for epoch in range(nepochs):
        train(model, train_loader, criterion, optimizer, epoch)
        scheduler.step()

    model.eval()
    data_list = test(model, test_loader)

    save_to_csv(data_list)
    print('done')
if __name__ == '__main__':
    main()
