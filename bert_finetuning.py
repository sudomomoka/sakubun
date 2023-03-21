import glob
import re
import os
import unicodedata
import string
import sys
import codecs
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from io import open
from sklearn.metrics import f1_score
from sklearn import datasets, linear_model
from gensim.models import KeyedVectors
import matplotlib

matplotlib.use('PS')
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from gensim.models import KeyedVectors
from torch.autograd import Variable
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit, KFold
from sklearn.metrics import accuracy_score
import itertools
from transformers import BertTokenizer, BertModel
from bert_process_functions import batched_index_select
from optimizer_Adamw import AdamW, LambdaLR, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR

print(torch.nn.utils.clip_grad_norm_)
print("cudnn version", torch.backends.cudnn.version())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)
#os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

# bert_model = BertModel.from_pretrained('./pytorch_model/', output_hidden_states=True)
# bert_model = bert_model.to(device)
tokenizer = BertTokenizer.from_pretrained('../../TemporalRelation/pytorch_model/')

best_acc_1 = 0
best_acc_2 = 0


# for name, param in bert_model.named_parameters():
#    param.requires_grad = True


def unicodeToUtf8(s):
    return s.encode('utf-8')


def readlines(filename):
    f = codecs.open(filename, 'r', 'utf-8')
    lines = f.read().strip().split('\n')
    f.close()
    return [unicodeToUtf8(line) for line in lines]


def read_multi_files(files_namelist):
    files_list = []
    for filename in files_namelist:
        # print(filename)
        each_file_line_list = readlines(filename)
        files_list += each_file_line_list
    # return [unicodeToUtf8(line) for line in lines if 'reltype' in line]
    #print(files_list)
    #exit()
    return files_list


def doc_kfold(data_dir, cv=5):
    file_list, file_splits = [], []
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".txt"):
            dir_file = os.path.join(data_dir, file)
            file_list.append(dir_file)
    #    logger.info("[Number] %i files in '%s'" % (len(file_list), data_dir))
    gss = KFold(n_splits=cv, shuffle=True, random_state=1029)
    for train_split, test_split in gss.split(file_list):
        file_splits.append((
            [file_list[fid] for fid in train_split],
            [file_list[fid] for fid in test_split]
        ))
    return file_splits


def padding_2d(seq_2d, max_len, pad_tok=0, direct='right'):
    for seq_1d in seq_2d:
        for i in range(0, max_len - len(seq_1d)):
            if direct in ['right']:
                seq_1d.append(pad_tok)
            else:
                seq_1d.insert(0, pad_tok)
    return seq_2d


def getSentAndCategory(datalist):
    global sent_list
    global category_list
    sent_list = []

    category_list = []
    for data in datalist:
        # sent_list.append((data[3],data[5]))
        if len(data) == 6:
            #print(data)
            sent_list.append((data[4], data[5]))
            category_list.append(data[0])
        if len(data) == 4:
            #print(data)
            sent_list.append((data[2],data[3]))
            category_list.append(data[0])
        #print(sent_list)
        #print(category_list)
        #exit()
    n_categories = len(set(category_list))
    # print(n_categories)
    return (sent_list, category_list, n_categories)


def process_data(train_path_list, test_path_list):
    datalist_train = [re.split('\t|\|', line.decode('utf-8')) for line in read_multi_files(train_path_list)]
    datalist_test = [re.split('\t|\|', line.decode('utf-8')) for line in read_multi_files(test_path_list)]

    #print(datalist_train)
    #print(datalist_train)
    #exit()

    sent_list_train, category_list_train, n_train_categories = getSentAndCategory(datalist_train)
    sent_list_test, category_list_test, n_test_categories = getSentAndCategory(datalist_test)
    sent_list_all, category_list, _ = getSentAndCategory(datalist_train + datalist_test)
    sent2category = {}
    for i in range(len(sent_list)):
        sent2category[sent_list[i]] = category_list[i]
    # print("sent:", sent_list_test[0][0].encode('utf-8').strip())
    # print(np.shape(sent_list_test[0]))
    x_train = np.array(sent_list_train)
    y_train = np.array(category_list_train)

    x_test = np.array(sent_list_test)
    y_test = np.array(category_list_test)

    cat2ix = get_cat2ix()
    doc_tensor_dict = get_sequence_2_TensorDataset(cat2ix, x_train, y_train, x_test, y_test)

    return cat2ix, doc_tensor_dict


def get_cat2ix():
    cat2ix = {}
    for cat in category_list:
        if cat not in cat2ix:
            cat2ix[cat] = len(cat2ix)
    print(cat2ix)
    return cat2ix


def seg_sent(x_data):
    seg_list = []
    for line in x_data:
        toks = mt.parse(line.strip()).splitlines()
        seg_list.append([tok.split('\t')[0] for tok in toks][:-1])
    return seg_list


def get_doc_2_tensor(cat2ix, x_train, y_train, x_test, y_test):
    global max_len
    tok_num = sum([len(line_tok) for line_tok in seg_sent(sent_list)])

    max_len = max([len(line_tok) for line_tok in seg_sent(sent_list)])

    unk_count = 0
    for line_tok in seg_sent(sent_list):
        for tok in line_tok:
            if tok not in word2ix:
                unk_count += 1
    print(unk_count, tok_num, unk_count / tok_num, max_len)

    x_train_t = x_train

    y_train_t = [torch.tensor([cat2ix[y]]) for y in y_train]
    x_test_t = x_test
    y_test_t = [torch.tensor([cat2ix[y]]) for y in y_test]
    doc_tensor_dict = {}

    doc_tensor_dict["x_train_t"] = x_train_t
    doc_tensor_dict["y_train_t"] = y_train_t
    doc_tensor_dict["x_test_t"] = x_test_t
    doc_tensor_dict["y_test_t"] = y_test_t

    return doc_tensor_dict


def get_doc_3_tensor(cat2ix, x_train, y_train, x_test, y_test):
    # x_train_t = x_train
    # x_train_t = x_train
    x_train_token_ids = []
    x_train_type_ids = []
    x_train_mask_ids = []

    x_test_token_ids = []
    x_test_type_ids = []
    x_test_mask_ids = []
    max_seq_length = 106
    from bert_process_functions import convert_tokens_to_features, convert_full_sequence_sdp_to_features
    train_features = convert_full_sequence_sdp_to_features(x_train, max_seq_length, tokenizer)
    test_features = convert_full_sequence_sdp_to_features(x_test, max_seq_length, tokenizer)
    # features_list  contains a list of feature dict

    # features.append( input_ids,input_mask,segment_ids )

    tr_input_ids = torch.tensor([d["input_ids"] for d in train_features])
    tr_input_mask = torch.tensor([d["input_mask"] for d in train_features])
    tr_position_mask = torch.tensor([d["position_mask"] for d in train_features])
    tr_type_ids = torch.tensor([d["type_ids"] for d in train_features])

    te_input_ids = torch.tensor([d["input_ids"] for d in test_features])
    te_input_mask = torch.tensor([d["input_mask"] for d in test_features])
    te_position_mask = torch.tensor([d["position_mask"] for d in test_features])
    te_type_ids = torch.tensor([d["type_ids"] for d in test_features])

    y_train_t = torch.tensor([cat2ix[y] for y in y_train])
    y_test_t = torch.tensor([cat2ix[y] for y in y_test])

    train_data = TensorDataset(tr_input_ids, tr_input_mask, tr_position_mask, tr_type_ids, y_train_t)
    test_data = TensorDataset(te_input_ids, te_input_mask, te_position_mask, te_type_ids, y_test_t)

    doc_tensor_dict = {
        "train_dataset": train_data,
        "test_dataset": test_data
    }

    return doc_tensor_dict


def get_sequence_2_TensorDataset(cat2ix, x_train, y_train, x_test, y_test):
    x_train_token_ids = []
    x_train_type_ids = []
    x_train_mask_ids = []

    x_test_token_ids = []
    x_test_type_ids = []
    x_test_mask_ids = []
    max_seq_length = 113
    from bert_process_functions import convert_tokens_to_features, convert_to_features
    # train_features  =  convert_tokens_to_features ( x_train   , max_seq_length, tokenizer )
    train_features = convert_to_features(x_train, max_seq_length, tokenizer)
    test_features = convert_to_features(x_test, max_seq_length, tokenizer)

    #print(train_features)
    #print(test_features)

    tr_input_ids = torch.tensor([d["input_ids"] for d in train_features])
    tr_input_mask = torch.tensor([d["input_mask"] for d in train_features])
    tr_type_ids = torch.tensor([d["type_ids"] for d in train_features])
    tr_position_mask = torch.tensor([d["position_mask"] for d in train_features])

    te_input_ids = torch.tensor([d["input_ids"] for d in test_features])
    te_input_mask = torch.tensor([d["input_mask"] for d in test_features])
    te_type_ids = torch.tensor([d["type_ids"] for d in test_features])
    te_position_mask = torch.tensor([d["position_mask"] for d in test_features])

    y_train_t = torch.tensor([cat2ix[y] for y in y_train])
    y_test_t = torch.tensor([cat2ix[y] for y in y_test])
    # print("y_test_t",y_test_t)
    train_data = TensorDataset(tr_input_ids, tr_input_mask, tr_position_mask, tr_type_ids, y_train_t)
    test_data = TensorDataset(te_input_ids, te_input_mask, te_position_mask , te_type_ids, y_test_t)

    doc_tensor_dict = {
        "train_dataset": train_data,
        "test_dataset": test_data
    }

    return doc_tensor_dict


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, concat_all=True, with_lstm=True):
        super(BiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        print("input_size : ", input_size)
        self.concat_all = concat_all
        self.with_lstm = with_lstm
        if self.concat_all:
            self.input_size = 4 * input_size
        else:
            self.input_size = 1 * input_size
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True,
                            dropout=0.3)
        if self.with_lstm:
            self.hidden = nn.Linear(hidden_size * 2, hidden_size * 2)
            self.out = nn.Linear(hidden_size * 2, output_size)

        else:
            if self.concat_all:
                self.hidden = nn.Linear(768 * 4, 200)
                self.out = nn.Linear(hidden_size * 1, output_size)

            else:
                self.hidden = nn.Linear(768, 200)
                self.out = nn.Linear(hidden_size * 1, output_size)

        self.bert_model = BertModel.from_pretrained('../../TemporalRelation/pytorch_model/', output_hidden_states=True)
        state_dict = torch.load("../../TemporalRelation/src/model/bccwj_model.pt")
        
        self.bert_model.load_state_dict(state_dict,strict=False)
        # bert_model = bert_model.to(device)
        self.freeze(False)

    def freeze(self, flag=False):
        if flag is True:
            for name, param in self.bert_model.named_parameters():
                if 'classifier' not in name:  # classifier layer
                    param.requires_grad = False
        else:
            for name, param in self.bert_model.named_parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, position_mask=None, token_type_ids=None, hidden_state=None):
        # with torch.no_grad():
        output_bert = self.bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        if self.concat_all:
            bert_emebddings = torch.cat(tuple([output_bert[2][i] for i in [-1, -2, -3, -4]]), dim=-1)

        else:
            bert_emebddings = output_bert[2][-2]  # second to last layer embeddings

        sdp_bert_embeddings = batched_index_select(bert_emebddings, 1, position_mask)
        # print ( "sdp_bert_embeddings shape ", sdp_bert_embeddings.shape   )
        if self.with_lstm:
            output, hidden_state = self.lstm(sdp_bert_embeddings.to(device), hidden_state)
            output, _ = torch.max(F.relu(output), 1)
        else:

            output, _ = torch.max(F.relu(sdp_bert_embeddings), 1)
        # print ( "max out ", output.shape )
        output = self.hidden(output)
        # print ( "hidden  out ", output.shape )

        output = self.out(F.relu(output))
        # print ( "relu  out ", output.shape )

        output = F.log_softmax(output, dim=-1)
        # print ( "log_softmax.shape ",  output.shape   )

        return output

    def initHidden(self, batch_size):
        h = torch.zeros(2, batch_size, self.hidden_size).to(device)
        c = torch.zeros(2, batch_size, self.hidden_size).to(device)
        return (h, c)


def define_model(len_cat2ix, concat_all=True, with_lstm=True):
    n_hidden = 200
    n_epoch = 20
    model = BiLSTM(768, n_hidden, len_cat2ix, concat_all=concat_all, with_lstm=with_lstm)
    model = model.to(device)

    return model


def train_model(n_epoch=10, train_loader=None, valid_loader=None, test_loader=None,
                model=None, optimizer=None, criterion=None,
                scheduler=None, flag=None):
    loss_list = []
    acc_list = []
    for epoch in range(1, n_epoch + 1):

        epoch_loss = []
        model.train()
        y_pred = []
        y_true = []
        best_loss_1 = 100000
        best_loss_2 = 100000

        for batch_data in tqdm(train_loader):
            batch_data = (e.to(device) for e in batch_data)
            input_bert_ids, input_mask, position_mask, token_type_ids, y = batch_data

            model.train()
            model.zero_grad()
            batch_size_ = input_bert_ids.shape[0]
            hidden_state = model.initHidden(batch_size_)

            # output = model(input_bert_ids.to(device), token_type_ids.to(device), hidden_state)
            output = model(input_bert_ids, input_mask, position_mask, token_type_ids, hidden_state)
            y = y.to(device)
            loss = criterion(output, y)
            loss.backward()
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pred = torch.argmax(output, dim=-1)

            y_true.append(y.detach().cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())
            epoch_loss.append(loss.item())

        y_true = np.array(list(itertools.chain(*y_true)))
        y_pred = np.array(list(itertools.chain(*y_pred)))
        accuracy = accuracy_score(y_true, y_pred)
        print('epoch %i, loss=%.4f, acc=%.2f%%' % (
            epoch, sum(epoch_loss) / len(epoch_loss), accuracy * 100))

        model.eval()
        with torch.no_grad():

            epoch_loss_val = []
            y_pred_val = []
            y_true_val = []
            for batch_data in valid_loader:
                batch_data = (e.to(device) for e in batch_data)
                input_bert_ids, input_mask, position_mask, token_type_ids, y = batch_data

                batch_size_ = input_bert_ids.shape[0]
                hidden_state = model.initHidden(batch_size_)

                # output = model(input_bert_ids.to(device), token_type_ids.to(device), hidden_state)
                output = model(input_bert_ids, input_mask, position_mask, token_type_ids, hidden_state)
                y = y.to(device)
                loss_val = criterion(output, y)
                pred = torch.argmax(output, dim=-1)
                y_true_val.append(y.detach().cpu().numpy())
                y_pred_val.append(pred.detach().cpu().numpy())
                epoch_loss_val.append(loss.item())

            y_true_val = np.array(list(itertools.chain(*y_true_val)))
            y_pred_val = np.array(list(itertools.chain(*y_pred_val)))
            acc_val = accuracy_score(y_true_val, y_pred_val)
            print('epoch %i, loss=%.4f, acc=%.2f%%' % (
                epoch, sum(epoch_loss_val) / len(epoch_loss_val), acc_val * 100))
            loss_list.append(sum(epoch_loss_val) / len(epoch_loss_val))
            acc_list.append(acc_val)

            #########
            # global best_loss_1, best_loss_2

            if loss_val < best_loss_1 and flag == 1:
                torch.save(model.state_dict(), '../model/model1.pt')
                best_loss_1 = loss_val

            if loss_val < best_loss_2 and flag == 2:
                torch.save(model.state_dict(), '../model/all_model.pt')
                best_loss_2 = loss_val

        model.eval()
        with torch.no_grad():
            y_pred = []
            y_true = []

            for batch_data in test_loader:
                batch_data = (e.to(device) for e in batch_data)
                input_bert_ids, input_mask, position_mask, token_type_ids, y = batch_data
                batch_size_ = input_bert_ids.shape[0]
                hidden_state = model.initHidden(batch_size_)

                output = model(input_bert_ids, input_mask, position_mask, token_type_ids, hidden_state)
                y = y.to(device)
                pred = torch.argmax(output, dim=-1)
                y_true.append(y.detach().cpu().numpy())
                y_pred.append(pred.detach().cpu().numpy())

            y_true = np.array(list(itertools.chain(*y_true)))
            y_pred = np.array(list(itertools.chain(*y_pred)))
            acc = accuracy_score(y_true, y_pred)
            from sklearn.metrics import f1_score
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_micro = f1_score(y_true, y_pred, average='micro')
            f1_weighted = f1_score(y_true, y_pred, average='weighted')
            results = {
                "acc": acc,
                "f1_macro": f1_macro,
                "f1_micro": f1_micro,
                "f1_weighted": f1_weighted,

            }
            print("------------------")
            print("test result", acc * 100)
            print("------------------")

    return loss_list, acc_list


def evaluate_model(model, test_loader):
    model.eval()

    with torch.no_grad():
        y_pred = []
        y_true = []

        for batch_data in test_loader:
            # print(batch_data)
            batch_data = (e.to(device) for e in batch_data)
            input_bert_ids, input_mask, position_mask, token_type_ids, y = batch_data
            batch_size_ = input_bert_ids.shape[0]
            hidden_state = model.initHidden(batch_size_)

            output = model(input_bert_ids, input_mask, position_mask, token_type_ids, hidden_state)
            y = y.to(device)
            pred = torch.argmax(output, dim=-1)

            y_true.append(y.detach().cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())
            # print("****************y_pred*********",np.shape(y_pred))

        y_true = np.array(list(itertools.chain(*y_true)))
        y_pred = np.array(list(itertools.chain(*y_pred)))
        acc = accuracy_score(y_true, y_pred)
        from sklearn.metrics import f1_score
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        results = {
            "acc": acc,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,

        }
        return results, acc


def create_data_loader_new(doc_tensor_dict, batch_size=16):
    train_dataset = doc_tensor_dict["train_dataset"]
    test_dataset = doc_tensor_dict["test_dataset"]
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    # val_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, test_loader


def create_data_loader(doc_tensor_dict, batch_size=16):
    train_dataset = doc_tensor_dict["train_dataset"]
    test_dataset = doc_tensor_dict["test_dataset"]
    # train_dataset, valid_dataset = split_hanshu(  train_dataset  )
    train_split = 0.85
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, valid_loader, test_loader


def InitialModel(n_epoch, len_cat2ix, train_loader, concat_all=True, with_lstm=True):
    model = define_model(len_cat2ix, concat_all=concat_all, with_lstm=with_lstm)
    # print("model parameters : ")
    max_grad_norm = 1.0
    num_training_steps = n_epoch * len(train_loader)
    num_warmup_steps = num_training_steps / 10

    param_optimizer = list(model.named_parameters())
    bert_name_list = ['bert_model']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in bert_name_list)], 'lr': 2e-5},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in bert_name_list)], 'lr': 2e-5}
    ]

    # To reproduce BertAdam specific behavior set correct_bias=False
    optimizer = AdamW(
        optimizer_grouped_parameters,
        correct_bias=False
    )

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    return model, optimizer, criterion, scheduler


"""
def fig_print(acc_list1 = None, acc_list2 = None, loss_list1 = None, loss_list2 = None, num_of_cv=None):

    x11 = range(0, 20)
    x12 = range(0, 20)
    x21 = range(0, 20)
    x22 = range(0, 20)
    y11 = acc_list1
    y12 = loss_list1
    y21 = acc_list2
    y22 = loss_list2
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x11, y11, '.-')
    plt.plot(x12, y21, 'x')
    plt.title('accuracy vs. epoches')
    plt.ylabel('accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x21, y12, '.-')
    plt.plot(x22, y22, 'x')
    plt.xlabel('loss vs. epoches')
    plt.ylabel('loss')
    plt.savefig("./fig/t2e/accuracy_loss_{}.jpg".format(num_of_cv))
"""


def main_cv(kfold_num=5):
    print("**** run main main_cv **** ")

    results_cv_list1 = list()
    results_cv_list2 = list()
    #data_dir = '小学生_all'
    #print("loaddatafile:", data_dir)
    data_splits = []
    #data_splits = doc_kfold(data_dir, cv=kfold_num)
    train_file = glob.glob("40-50_all/*.*")
    test_file = glob.glob("60-80_all/*.*")
    data_splits.append((train_file,test_file))

    #print(data_splits)
    #exit()

    acc_all1 = []
    acc_all2 = []
    for cv_index in range(kfold_num):
        if cv_index < -4:
            continue
        else:

            print("run the ")
            train_path = data_splits[cv_index][0]
            test_path = data_splits[cv_index][1]

            train_path_list = data_splits[cv_index][0]
            test_path_list = data_splits[cv_index][1]
            print(len(train_path_list), train_path_list)
            print(len(test_path_list), test_path_list)

            # get_dataloader
            cat2ix, doc_tensor_dict = process_data(train_path_list, test_path_list)
            train_loader, valid_loader, test_loader = create_data_loader(doc_tensor_dict, batch_size=16)
            print("length of train_loader:", len(train_loader))

            # define model and optimizer
            n_epoch = 20

           # model1, optimizer1, criterion1, scheduler1 = InitialModel(n_epoch, len(cat2ix), train_loader,
            #                                                          concat_all=False, with_lstm=True)

            model2, optimizer2, criterion2, scheduler2 = InitialModel(n_epoch, len(cat2ix), train_loader,
                                                                      concat_all=False, with_lstm=False)

             #======== Train model    =====================#

            """
            loss_list1, acc_list1 = train_model(n_epoch, train_loader=train_loader, valid_loader=valid_loader,
                                                test_loader=test_loader,
                                                model=model1,
                                                optimizer=optimizer1,
                                                criterion=criterion1,
                                                scheduler=scheduler1, flag=1)"""


            loss_list2, acc_list2 = train_model(n_epoch, train_loader=train_loader, valid_loader=valid_loader,
                                                test_loader=test_loader, model=model2,
                                                optimizer=optimizer2,
                                                criterion=criterion2,
                                                scheduler=scheduler2, flag=2)

            # fig_print(acc_list1=acc_list1, acc_list2=acc_list2,
            #      loss_list1=loss_list1, loss_list2=loss_list2,
            #      num_of_cv=cv_index)

            # ======== evaluate  model    =====================#

            #model1.load_state_dict(torch.load('../model/model1_yes.pt'))
            model2.load_state_dict(torch.load('../model/all_model.pt'))
            #results1, acc_1 = evaluate_model(model1, test_loader)
            results2, acc_2 = evaluate_model(model2, test_loader)
            #acc_all1.append(acc_1)
            acc_all2.append(acc_2)
            #results_cv_list1.append(results1)
            results_cv_list2.append(results2)

    #df1 = pd.DataFrame(results_cv_list1)
    df2 = pd.DataFrame(results_cv_list2)
    print(" results cv is : ")
    #print(df1)
    print(df2)
    #df1.to_csv('result_cv.tsv1', float_format='%.2f', sep='\t')
    df2.to_csv('result_cv.tsv2', float_format='%.2f', sep='\t')
    #print("acc_average yes:", sum(acc_all1) / kfold_num)
    print("acc_average no:", sum(acc_all2) / kfold_num)


def main():
    main_cv(kfold_num=5)


if __name__ == "__main__":
    main()
