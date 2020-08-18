# -*- coding: utf-8 -*-
import torch
import pandas as pd
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pynlpir
import random
import torch.nn.utils.rnn as rnn
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#~~~~~~~~~~~~~~~settings change between gpu(linux) & cpu(windows)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#mode = 'cpu'
mode = 'cuda'

if mode == 'cpu':
    torch.set_default_tensor_type(torch.FloatTensor)
    path = ''
    cmd = 'del' #windows os
if mode == 'cuda':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    path = '/content/gdrive/My Drive/Colab Notebooks/' # altered due to the server's file path ALTER WHEN CHANGE
    cmd = 'rm' #linux os

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#network architecture  (tbd)
class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, p, n):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, dropout=p, num_layers = n)
        self.hidden2label = nn.Linear(2*hidden_dim, label_size)
        self.num_layer = n
        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (autograd.Variable(torch.zeros(2*self.num_layer, self.batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(2*self.num_layer, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        #self.lstm1.flatten_parameters()
        #self.lstm2.flatten_parameters()
        #self.lstm3.flatten_parameters()
        embeds = self.word_embeddings(sentence)
        #print(embeds.shape)
        a = embeds.clone().view(len(sentence), self.batch_size, -1)
        #print(a.shape)
        lstm_out, self.hidden = self.lstm(a, self.hidden)
        temp = (lstm_out[-1]).clone()
        y = self.hidden2label(temp)
        log_probs = F.log_softmax(y, dim = 1)
        #log_probs = F.softmax(y, dim = 1)
        return log_probs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right/len(truth)


def train():
    # opening embedding file
    print('opening embedding file')
    f = open(path + 'sgns.weibo.bigram-char', 'r', encoding='utf8')
    raw = f.readlines()
    f.close()

    # constructing word to index dictionary
    print('constructing word to index dictionary')
    word_to_ix = dict()
    iter = 0
    for line in raw:
        word_to_ix[(line.split())[0]] = iter
        iter = iter + 1

    for i in ['ttttt', 'ggggg', 'uuuuu', 'eeeee', 'ooooo', ' ']:
        word_to_ix[i] = iter
        iter += 1

    # loading the pre-trained embedding vectors
    print('loading the pre-trained embedding vectors')
    embed_vectors = []
    for line in raw:
        embed_vectors.append([float(j) for j in ((line.split())[1:])])

    for i in ['ttttt', 'ggggg', 'uuuuu', 'eeeee', 'ooooo']:
        embed_vectors.append((torch.zeros(300)).tolist()) #randn or zeros

    embed_vectors.append((torch.zeros(300)).tolist()) #for ' '
    global FINAL
    FINAL = len(embed_vectors) - 1

    # load the train, val, test data
    print('load the train, val, test data')
    df = pd.read_excel(path + 'final data.xlsx')

    # generate matrix with first-col uid second-col botornot
    idbotmat = []
    for i in range(len(df)):
        temp = []
        temp.append(df['uid'][i])
        temp.append(df['botornot'][i])
        idbotmat.append(temp)

    # all text into text_set
    # where every element is [sentence, label]
    # sentence is another list
    text_set = []
    pynlpir.open()  # 打开分词器
    for pair in idbotmat:
        f = open(path + 'data/' + str(pair[0]) + '.txt', 'r', encoding='utf8')
        #print(f)
        raw = f.readlines()
        f.close()
        if len(raw) == 0:
          continue

        for line in raw:
            # tokenizer from chinese academy of sciences
            temp = pynlpir.segment(line, pos_tagging=False)  # 使用pos_tagging来关闭词性标注
            temp2 = [x for x in temp if x != ' '] #remove redundant spaces

            sentence = []
            for word in temp2:
                if word in word_to_ix.keys():
                    sentence.append(word_to_ix[word])
                else:
                    sentence.append(word_to_ix['ooooo']) #oov

            data = [sentence, pair[1]]
            text_set.append(data)
    pynlpir.close()
    print('Total sentences: ' + str(len(text_set)))
    random.shuffle(text_set)

    #determine train dev test
    train_ratio = 0.7
    dev_ratio = 0.1
    test_ratio = 0.2
    total_sample = len(text_set)
    train_set = text_set[:int(total_sample * train_ratio)]
    dev_set = text_set[int(total_sample * train_ratio) : int(total_sample * (train_ratio + dev_ratio))]
    test_set = text_set[int(total_sample * (train_ratio + dev_ratio)):]

    print('delcaring model')
    best_dev_acc = 0.0
    EMBEDDING_DIM = len(embed_vectors[0])
    VOCAB_SIZE = len(embed_vectors)
    HIDDEN_DIM = 100 # TO BE TUNED
    LABEL_SIZE = 2
    BATCH_SIZE = 64 # TO BE TUNED
    EPOCH = 100 # TO BE TUNED
    DROPOUT = 0.5 # TO BE TUNED 0.95* every epoch
    NUM_LAYER = 2 # TO BE TUNED

    # declare model
    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, LABEL_SIZE, BATCH_SIZE, DROPOUT, NUM_LAYER)

    #load the word embedding
    embedtensor = torch.tensor(embed_vectors)
    #embedtensor.to(device)
    model.word_embeddings.weight.data = embedtensor
    #model.to(device)

    #how many parameters in the model
    print('Total params in the network: ' + str(count_parameters(model)))

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-2) # TO BE TUNED
    #optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum = 0.9)

    print('training starts now')
    torch.autograd.set_detect_anomaly(True)
    no_up = 0
    for i in range(EPOCH):
        print('epoch: %d start!' % i)
        random.shuffle(train_set)
        model.lstm.dropout = DROPOUT
        train_epoch(model, train_set, loss_function, optimizer, BATCH_SIZE, i) #what to do? what to pass?
        DROPOUT = DROPOUT * 0.95 #dropout scheduling
        print('now best dev acc:', best_dev_acc)
        dev_acc = evaluate(model, dev_set, loss_function, 'dev') #what to do? what to pass?
        test_acc = evaluate(model, test_set, loss_function, 'test') #what to do? what to pass?
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            #!rm '/content/gdrive/My Drive/Colab Notebooks/best_models/mr_best_model_minibatch_acc_*.model' #colab only
            os.system(cmd + ' ' + path +  'best_models/mr_best_model_minibatch_acc_*.model')
            print('New Best Dev!!!')
            torch.save(model.state_dict(),path + 'best_models/mr_best_model_minibatch_acc_' + str(int(test_acc * 10000)) + '.model')
            no_up = 0
        else:
            no_up += 1
            if no_up >= 10:
              print("so what")
              #sys.exit()
              #exit()

def evaluate(model, eval_set, loss_function, name = 'dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    model.batch_size = 1
    for pair in eval_set:
        truth_res.append(pair[1])
        model.hidden = model.init_hidden() #really? detach it from last instance
        temp = torch.tensor(pair[0])
        #temp.to(device)
        pred = model(temp)
        pred_res.append(pred.detach().cpu().numpy().argmax())
        #print(pred)
        #print(pred.unsqueeze(0))
        #print(torch.tensor(pair[1]).unsqueeze(0))
        templabel = torch.tensor(pair[1]).unsqueeze(0)
        #templabel.to(device)
        loss = loss_function(pred, templabel)
        avg_loss += loss.item()

    avg_loss /= len(eval_set)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss:%g train acc:%g' % (avg_loss, acc))
    return acc

def train_epoch(model, train_set, loss_function, optimizer, batch_size, i):
    model.train()
    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    batch_start = 0
    batch_end = batch_size
    while True:
        #print("haha")
        to_pad = []
        label = []
        for pair in train_set[batch_start:batch_end]:
            temp = (torch.tensor(pair[0])).clone()
            to_pad = to_pad + [temp]
            #to_pad.append(temp)
            truth_res = truth_res + [pair[1]]
            #truth_res.append(pair[1])
            label = label + [pair[1]]
            #label.append(pair[1])
        real_to_pad = to_pad
        batch_sent = rnn.pad_sequence(real_to_pad, padding_value= FINAL, batch_first = False)
        #batch_sent.to(device)
        model.batch_size = (batch_end - batch_start)
        model.hidden = model.init_hidden() #really? detach from last instance
        #print(batch_sent)
        pred = model(batch_sent)
        for res in pred:
            temp = res.clone()
            pred_res = pred_res + [temp.detach().cpu().numpy().argmax()]
            #pred_res.append(temp.detach().cpu().numpy().argmax())
        model.zero_grad()
        labels = (torch.tensor(label)).clone()
        labels.requires_grad = False
        #labels.to(device)
        loss = loss_function(pred, labels)
        #print(pred)
        #print(labels)
        #print(loss.item())
        #loss.to(device)
        avg_loss = avg_loss + loss.item()
        count = count + 1
        if count % 100 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count * model.batch_size, loss.item()))
        loss.backward(retain_graph = True)
        optimizer.step()
        #print(model.word_embeddings.weight.data[0]) #validate the change of embedding

        #update batch start/end
        if batch_end == len(train_set):
            break
        batch_start += batch_size
        batch_end += batch_size
        if batch_end > len(train_set):
            batch_end = len(train_set)
    avg_loss /= (len(train_set) / batch_size)
    print('epoch: %d done!\ntrain avg_loss:%g , acc:%g' % (i, avg_loss, get_accuracy(truth_res, pred_res)))

FINAL = 0
train()
