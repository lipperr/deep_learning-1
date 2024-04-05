import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm.autonotebook import tqdm
# from pathlib import Path
import os

from dataset import TextDataset
from model import *
from train import Trainer, SaveModel

import warnings
warnings.filterwarnings('ignore')

import gc

# Path('/kaggle/working/model').mkdir(parents=True, exist_ok=True)




### вся работа c помощью https://pytorch.org/tutorials/beginner/translation_transformer.html

def main():

    cwd = os.getcwd()

    # data_dir = '/kaggle/input/bhw-2-translation-dataset/bhw2-data/' 
    data_dir = cwd + '/bhw-2-translation/data/'
    train_pth = 'train.de-en.'
    val_pth = 'val.de-en.'
    test_pth = 'test1.de-en.'
    src_lang = 'de'
    tgt_lang = 'en'

    num_encoder_layers = 3
    num_decoder_layers = 3
    emb_size = 512
    nhead = 8
    dim_feedforward = 512

    batch_size = 128
    vocab_size = 30000
    n_epochs = 10

    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(SEED)
    torch.cuda.empty_cache()
    gc.collect()
    
    wandb.init(project='bhw-2')

    train_dataset = TextDataset('train', data_dir+train_pth+src_lang, data_dir+train_pth+tgt_lang, vocab_size=vocab_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate)

    val_dataset = TextDataset('val', data_dir+val_pth+src_lang, data_dir+val_pth+src_lang, vocab_size=vocab_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_dataset.collate)


    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")


    transformer = Seq2SeqTransformer(num_encoder_layers, num_decoder_layers, emb_size, nhead, vocab_size, vocab_size, dim_feedforward)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(device)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

    model_saver = SaveModel()
    trainer = Trainer(transformer, device, optimizer, criterion, model_saver)
    trainer.train(train_loader, val_loader, n_epochs)

    test_dataset = TextDataset('test', data_dir+test_pth+src_lang, vocab_size=vocab_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate)

    translated = []

    for src in tqdm(test_loader):
        sentence = translate(trainer.model, src, device, train_dataset.token_transform[tgt_lang].decode)
        translated.append(sentence)

    with open('/kaggle/working/output.txt', "w") as f:
        for sentence in translated:
            f.write(sentence + "\n")

    wandb.finish()

if __name__ == "__main__":
    main()