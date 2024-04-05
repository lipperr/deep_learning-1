import os
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.nn.utils.rnn import pad_sequence
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
import numpy as np



UNK_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
PAD_IDX = 3


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


def yield_tokens(tokenizer, texts):
    for text_sample in tqdm(texts, desc="Building vocab"):
        sentence = tokenizer(text_sample)
        for token in sentence:
            yield token



class TextDataset(Dataset):

    def __init__(self, type, path_src, path_tgt='', path_vocab='vocabs', vocab_size=50000):
        self.type = type
        self.path_src = path_src
        self.lang_src = 'de'
        self.path_tgt = path_tgt
        self.lang_tgt = 'en'

        self.vocab_size = vocab_size

        self.texts = {'de':[], 'en':[]}
        with open(self.path_src, encoding="utf-8") as f:
            self.texts[self.lang_src] = [line.rstrip() for line in f.readlines()]
        
        if self.type != 'test':
            with open(self.path_tgt, encoding="utf-8") as f:
                self.texts[self.lang_tgt] = [line.rstrip() for line in f.readlines()]

        if self.type != 'test':
            assert len(self.texts[self.lang_src]) == len(self.texts[self.lang_tgt]), "Size of src and dst datasets must match"


        self.token_transform = {}

        if not os.path.isfile('tokenizer_src' + '.model'):
            SentencePieceTrainer.train(
                input=self.path_src, vocab_size=self.vocab_size,
                model_type='word', model_prefix='tokenizer_src',
                normalization_rule_name='nmt_nfkc_cf',
                pad_id = PAD_IDX
            )
        self.token_transform[self.lang_src] = SentencePieceProcessor(model_file='tokenizer_src' + '.model')

        if self.type != 'test':
            if not os.path.isfile('tokenizer_tgt' + '.model'):
                SentencePieceTrainer.train(
                    input=self.path_tgt, vocab_size=self.vocab_size,
                    model_type='word', model_prefix='tokenizer_tgt',
                    normalization_rule_name='nmt_nfkc_cf',
                    pad_id = PAD_IDX
                )
            self.token_transform[self.lang_tgt] = SentencePieceProcessor(model_file='tokenizer_tgt' + '.model')

            
        self.text_transform = {}
        
        self.text_transform[self.lang_src] = sequential_transforms(
            self.token_transform[self.lang_src].encode, TextDataset.tensor_transform)
        
        if self.type != 'test':
            self.text_transform[self.lang_tgt] = sequential_transforms(
                self.token_transform[self.lang_tgt].encode, TextDataset.tensor_transform)

    @staticmethod
    def tensor_transform(token_ids):
        return torch.cat((torch.tensor([BOS_IDX]),
                        torch.tensor(token_ids),
                        torch.tensor([EOS_IDX])))
    
    def __len__(self):
        return len(self.texts[self.lang_src])


    def __getitem__(self, index):
        if self.type != 'test':
            return tuple([self.text_transform[self.lang_src](self.texts[self.lang_src][index]), self.text_transform[self.lang_tgt](self.texts[self.lang_tgt][index])])
        return tuple([self.text_transform[self.lang_src](self.texts[self.lang_src][index])])
    
    def collate(self, batch):
        maxlen = 0
        
        if self.type != 'test':
            maxlen = max([max(batch[i][0].shape[0], batch[i][1].shape[0]) for i in range(len(batch))])
        else:
            maxlen = max([batch[i][0].shape[0] for i in range(len(batch))])

        src_batch = torch.Tensor([torch.cat((b[0], torch.full((maxlen - b[0].shape[0], ), fill_value=3))).numpy() for b in batch])


        if self.type == 'test':
            return src_batch.T

        tgt_batch = torch.Tensor([torch.cat((b[1], torch.full((maxlen - b[1].shape[0], ), fill_value=3))).numpy() for b in batch])


        return src_batch.T, tgt_batch.T
