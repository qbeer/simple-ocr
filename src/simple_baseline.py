import torch
import lightning.pytorch as pl
from torchmetrics.text import WordErrorRate, CharErrorRate

class BaselineModel(pl.LightningModule):
    def __init__(self, n_chars, seq_len, idx2char):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(1, 2)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2)),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(512, seq_len*n_chars),
            torch.nn.Unflatten(1, unflattened_size=(seq_len, n_chars))     
        )
        
        self.seq_len = seq_len
        self.n_chars = n_chars
        
        self.idx2char = idx2char
        
        self.wer = WordErrorRate()
        self.cer = CharErrorRate()
        
        self.valid_wer = WordErrorRate()
        self.valid_cer = CharErrorRate()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x)
        bs = x.shape[0]
        loss = torch.nn.functional.cross_entropy(y_logits.view(-1, self.seq_len * self.n_chars),
                                                 y.view(-1, self.seq_len * self.n_chars), reduction='sum') / bs
        
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        
        # Decode predicted and true indices to characters
        pred_decoded = self.__decode(y_logits)
        true_decoded = self.__decode(y)
        
        word_error_rate = self.wer(pred_decoded, true_decoded)
        char_error_rate = self.cer(pred_decoded, true_decoded)
        
        self.log('train_wer', word_error_rate, prog_bar=True)
        self.log('train_cer', char_error_rate, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x)
        bs = x.shape[0]
        loss = torch.nn.functional.cross_entropy(y_logits.view(-1, self.seq_len * self.n_chars),
                                                 y.view(-1, self.seq_len * self.n_chars), reduction='sum') / bs
        
        self.log('valid_loss', loss, prog_bar=True)
        
        # Decode predicted and true indices to characters
        pred_decoded = self.__decode(y_logits)
        true_decoded = self.__decode(y)
        
        word_error_rate = self.valid_wer(pred_decoded, true_decoded)
        char_error_rate = self.valid_cer(pred_decoded, true_decoded)
        
        self.log('valid_wer', word_error_rate, prog_bar=True)
        self.log('valid_cer', char_error_rate, prog_bar=True)
        
        return loss
    
    def __decode(self, y):
        y = torch.argmax(y, dim=-1)
        y = y.detach().cpu().numpy()
        decoded = [ ''.join([ self.idx2char[idx] for idx in sample ]) for sample in y ]
        return decoded
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)