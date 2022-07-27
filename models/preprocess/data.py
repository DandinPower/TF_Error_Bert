import collections
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
load_dotenv()
MODEL_PATH = os.getenv('PRETRAIN_DIR_PATH')
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))

def GetTrainDataset(datasetPath,maxLen,splitRate, batchSize):
    print('Loading Train data....')
    vocab = load_vocab()
    dataset = YelpDataset(datasetPath, maxLen, vocab, splitRate)
    train_data = dataset.GetTrain()
    train_loader = DataLoader(train_data[0], train_data[1], batchSize, False)
    return train_loader.GetBatchDataset()

def GetTestDataset(datasetPath,maxLen,splitRate, batchSize):
    print('Loading Test data....')
    vocab = load_vocab()
    dataset = YelpDataset(datasetPath, maxLen, vocab, splitRate)
    test_data = dataset.GetTest()
    test_loader = DataLoader(test_data[0], test_data[1], batchSize, True)
    return test_loader.GetBatchDataset()

def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def load_vocab():
    data_dir = MODEL_PATH
    vocab = Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir,
        'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    return vocab

class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs

class YelpDataset():
    def __init__(self, datasetPath, max_len, vocab, splitRate):
        self.max_len = max_len
        self.labels = []
        self.vocab = vocab
        self.all_tokens_ids = []
        self.all_segments = []
        self.valid_lens = []
        self.all_datas = []
        self.path = datasetPath
        self.splitRate = splitRate
        self.Preprocess()
    
    #將資料做預處理
    def Preprocess(self):
        texts,self.labels = self.ReadDataset()
        texts = [self.TruncatePairOfTokens(text)for text in texts]
        newTexts,newSegments = [],[]
        for text in texts:
            tokens,segments = self.GetTokensAndSegments(text)
            newTexts.append(tokens)
            newSegments.append(segments)
        self.PadBertInput(newTexts, newSegments)
        self.Merge()

    #讀取dataset
    def ReadDataset(self):
        df = pd.read_csv(self.path)
        labels = []
        texts = []
        for i in range(len(df.Stars.values)):
            text = df.Text.values[i]
            label = df.Stars.values[i]
            if (type(text) != str): continue
            if label >= 4:
                labels.append(1)
            else:
                labels.append(0)
            texts.append(text.strip().lower().split(' '))
        self.trainLen = int(len(df.Text.values) * self.splitRate) 
        return texts,labels

    def GetTokensAndSegments(self,tokensA, tokensB=None):
        tokens = ['<cls>'] + tokensA + ['<sep>']
        # 0 and 1 are marking segment A and B, respectively
        segments = [0] * (len(tokensA) + 2)
        if tokensB is not None:
            tokens += tokensB + ['<sep>']
            segments += [1] * (len(tokensB) + 1)
        return tokens, segments

    #給<CLS>,<SEP>,<SEP>保留位置
    def TruncatePairOfTokens(self, tokens):   
        while len(tokens) > self.max_len - 3:
            tokens.pop()
        return tokens

    #進行padding
    def PadBertInput(self,texts,segments):
        texts = self.vocab[texts]
        for (text,segment) in zip(texts,segments):
            paddingText = np.array(text + [self.vocab['<pad>']] * (self.max_len - len(text)), dtype=np.long)
            self.all_tokens_ids.append(paddingText)
            self.all_segments.append(np.array(segment + [0] * (self.max_len - len(segment)), dtype=np.long))
            #valid_lens不包括<pad>
            self.valid_lens.append(np.array(len(text), dtype=np.float32))

    def Merge(self):
        self.all_tokens_ids = tf.constant(self.all_tokens_ids)
        self.all_segments = tf.constant(self.all_segments)
        self.valid_lens = tf.constant(self.valid_lens)
        for i in range(len(self.all_tokens_ids)):
            self.all_datas.append((self.all_tokens_ids[i],self.all_segments[i],self.valid_lens[i]))

    def GetTrain(self):
        return self.all_datas[0:self.trainLen],self.labels[0:self.trainLen]

    def GetTest(self):
        return self.all_datas[self.trainLen:],self.labels[self.trainLen:]

    def __len__(self):
        return len(self.all_tokens_ids)

class DataLoader():
    def __init__(self,datas,labels,batch,shuffle):
        self.datas = datas
        self.labels = labels
        self.batch = batch 
        self.shuffle = shuffle 
        self.start = 0
        self.turns = len(self.datas) // self.batch
        self.inputs = []
        self.PreLoad()

    #將資料集讀取進來
    def PreLoad(self):
        for i in range(len(self.datas)):
            self.inputs.append(self.datas[i][0])

    #回傳dataset
    def GetBatchDataset(self):
        bufferSize = 1
        if (self.shuffle):
            bufferSize = len(self.datas)
        dataset = tf.data.Dataset.from_tensor_slices((self.inputs, self.labels)).batch(self.batch).shuffle(buffer_size = bufferSize)
        return dataset

