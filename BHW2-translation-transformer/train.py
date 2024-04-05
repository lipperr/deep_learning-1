from tqdm.autonotebook import tqdm
import torch
import gc
import wandb
from model import create_mask

class Trainer:
    def __init__(self, model, device, optimizer, criterion, model_saver):
        self.model = model
        self.model = self.model.to(device)
        self.device = device

        self.optimizer = optimizer
        self.criterion = criterion
        
        self.model_saver = model_saver


    def train_epoch(self, dataloader, desc):
        self.model.train()
        losses = 0
        
        for src, tgt in tqdm(dataloader, desc):

            src = src.type(torch.long)
            tgt = tgt.type(torch.long)

            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, self.device)

            logits = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            self.optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            self.optimizer.step()
            losses += loss.item()
            
        return losses / len(list(dataloader))


    def evaluate(self, dataloader):
        self.model.eval()
        losses = 0

        for src, tgt in dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = self.model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / len(list(dataloader))



    def train(self, train_loader, val_loader, n_epochs, start_epoch=0, continue_training=False, model_path='model/saved_model.pth', log=True):
        if continue_training:
            start_epoch = self.load_model(model_path)
        for epoch in range(start_epoch, n_epochs):
            
            torch.cuda.empty_cache()
            gc.collect()
            
            train_loss = self.train_epoch(train_loader, f'Training epoch {epoch}/{n_epochs}')
            val_loss = self.train_epoch(val_loader, f'Validating epoch {epoch}/{n_epochs}')
            
            if log:
                wandb.log({'train_loss':train_loss, 'val_loss':val_loss})

            print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
            self.model_saver(val_loss, epoch, self.model, self.optimizer, model_path)
            
            
    def load_model(self, model_path='model/saved_model.pth'):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
        
        
        
        
class SaveModel:
    def __init__(self, save_best=False, best_val_loss=torch.inf):
        self.save_best = save_best
        self.best_val_loss = best_val_loss
        
    def __call__(self, val_loss, epoch, model, optimizer, model_path='model/saved_model.pth'):
        
        if val_loss < self.best_val_loss or not self.save_best:
            self.best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, model_path)
            print('New best model with loss {:.5f} is saved'.format(val_loss))
