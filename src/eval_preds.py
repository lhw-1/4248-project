import os

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

from models import InferSent

glove_path = '../GloVe/glove.840B.300d.txt'
#specify filepath for predictions (relative to src)
valid_filepath = '../dataset/esnli_dev.csv'
#specify filepath for trained InferSent model
infersent_filepath = '../savedir/model78.pickle'
#csv saved '../pred_outputs/output.csv'

def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec

def get_glove(word_dict, glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            #print(line)
            #break
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}(/{1}) words with glove vectors'.format(
                len(word_vec), len(word_dict)))
    return word_vec

def get_word_dict(sentences):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in str(sent).split():
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict

class NLINet(nn.Module):
    def __init__(self, config):
        super(NLINet, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        self.encoder = model  #eval(self.encoder_type)(config)
        self.inputdim = 4*2*self.enc_lstm_dim
        self.inputdim = 4*self.inputdim if self.encoder_type in \
                        ["ConvNetEncoder", "InnerAttentionMILAEncoder"] else self.inputdim
        self.inputdim = self.inputdim/2 if self.encoder_type == "LSTMEncoder" \
                                        else self.inputdim
        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
                )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
                )

    def forward(self, s1, s2):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)
        v = self.encoder(s2)

        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        output = self.classifier(features)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb

def get_batch(batch, word_vec, emb_dim=300):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), emb_dim))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]

    return torch.from_numpy(embed).float(), lengths

def evaluate_preds(nli_net, input_filepath=valid_filepath, output_filename='output.csv', final_eval=True):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop
    
    input = pd.read_csv(input_filepath, usecols=['gold_label', 'Sentence1', 'Sentence2'])
    #input.rename(columns={'Sentence1':'s1', 'Sentence2':'s2'}, inplace=True)
    #print(input.columns)
    #map label to int
    label_to_int = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    target = input['gold_label'].apply(lambda x: label_to_int[x]).tolist()
    
    s1 = input['Sentence1'].tolist()
    s2 = input['Sentence2'].tolist()
    s1 = np.array([
                    ['<s>'] + \
                    [word for word in str(sent).split() if word in word_vec] + \
                    ['</s>'] for sent in s1
                  ])
    s2 = np.array([
                    ['<s>'] + \
                    [word for word in str(sent).split() if word in word_vec] + \
                    ['</s>'] for sent in s2
                  ])

    #for generation of csv with predictions
    preds = []
    
    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        
        preds.extend(output.data.max(1)[1].detach().cpu().tolist())

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    #save csv file
    if not os.path.exists('../pred_outputs'):
                os.makedirs('../pred_outputs')
    pd.DataFrame({'label': target, 'Sentence1': s1, 'Sentence2': s2, 'prediction': preds}).to_csv(
        '../pred_outputs/'+output_filename, index=False
    )
        
    # save model
    eval_acc = 100 * correct/len(s1)  #round(100 * correct / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy: {0}'.format(eval_acc))
    else:
        print('togrep : results : mean accuracy:\
              {1}'.format(eval_acc))

    return eval_acc

# Load model
model_version = 1
MODEL_PATH = "../encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# Keep it on CPU or put it on GPU
use_cuda = torch.cuda.is_available()
#or force not to use cuda
#use_cuda = False
model = model.cuda() if use_cuda else model

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = '../GloVe/glove.840B.300d.txt' if model_version == 1 else '../fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

valid = pd.read_csv(valid_filepath, usecols=['gold_label', 'Sentence1', 'Sentence2'])
#valid.info()

#map label to int
label_to_int = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

valid['label'] = valid['gold_label'].apply(lambda x: label_to_int[x])

#converts DataFrames to dict
valid = valid.to_dict(orient='list')

word_vec = build_vocab(valid['Sentence1'] + valid['Sentence2'], glove_path)


parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--nlipath", type=str, default='dataset/SNLI/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--word_emb_path", type=str, default="dataset/GloVe/glove.840B.300d.txt", help="word embedding file path")

# training
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=128)  #64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='InferSentV1', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# gpu
parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")

params, _ = parser.parse_known_args()
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,

}
nli_net = NLINet(config_nli_model)

# Run best model on test set.
nli_net.load_state_dict(torch.load(infersent_filepath))

nli_net.to('cuda')

#print('\nTEST : Epoch {0}'.format(epoch))
#evaluate(1e6, 'valid', True)
#evaluate(0, 'test', True)

evaluate_preds(nli_net)