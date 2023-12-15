import torch
import lightning.pytorch as pl
from torchmetrics.text import WordErrorRate, CharErrorRate

class SeqModel(pl.LightningModule):
    def __init__(self, n_chars, seq_len, idx2char, teacher_forcing=False):
        super(SeqModel, self).__init__()
        self.feature_net = torch.nn.Sequential(
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
        )
        
        self.lstm = torch.nn.LSTM(input_size=n_chars, hidden_size=512,
                                  proj_size=n_chars,
                                  num_layers=2, batch_first=True,
                                  bidirectional=False)
        
        self.seq_len = seq_len
        self.n_chars = n_chars
        
        self.idx2char = idx2char
        
        self.teacher_forcing = teacher_forcing
        
        self.wer = WordErrorRate()
        self.cer = CharErrorRate()
        
        self.valid_wer = WordErrorRate()
        self.valid_cer = CharErrorRate()
        
    def forward(self, x, y):
        
        feat = self.feature_net(x) # (bs, 512)
        # NOTE: both h0 and c0 are initialized by the feature network
        #       these are the initial hidden and cell states of the LSTM
        c = feat.unsqueeze(1) # (bs, 1, 512) -> seq_len = 1
        c = c.permute(1, 0, 2) # (1, bs, 512)
        c = c.repeat(self.lstm.num_layers, 1, 1) # (4, bs, 512)
        
        h = torch.zeros((self.lstm.num_layers, feat.shape[0], self.lstm.proj_size)).to(feat.device)
        
        outputs = []
        
        # NOTE: we are using the start token as the first input to the LSTM
        #       this is a common practice in sequence-to-sequence models
        #       the start token is not learnable
        token = torch.zeros((h.shape[1], 1, self.n_chars)).to(h.device) # non-learnable start token
        
        #print(token.shape, h.shape, c.shape)
        
        for ind in range(self.seq_len):
            token, (h, c) = self.lstm(token, (h, c))
            outputs.append(token.squeeze(1))
            token = token.softmax(dim=-1)
            if self.teacher_forcing:
                token = y[ind, :] # in first loop this is the first true token
            #print(token.shape, h.shape, c.shape)
        
        outputs = torch.stack(outputs, dim=0) # (seq_len, bs, n_chars)
        outputs = outputs.permute(1, 0, 2) # (bs, seq_len, n_chars)
        
        #print(outputs[:5])
        
        return outputs
    
    def training_step(self, batch, batch_idx):
        x, y = batch # y -> (bs, 2, 10)
        y_logits = self.forward(x, y) # (bs, 2, 10)
        bs = x.shape[0] # bs
        # cross entropy expects logits: (bs, C), (bs,)
        # in our case (bs, 2, 10) -> (bs * 2, 10)
        _y = y.reshape(bs * self.seq_len, self.n_chars)
        _y_logits = y_logits.reshape(bs * self.seq_len, self.n_chars)
        loss = torch.nn.functional.cross_entropy(_y_logits, _y, reduction='sum') / bs
        
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
        y_logits = self.forward(x, y)
        bs = x.shape[0]
        
        _y = y.reshape(bs * self.seq_len, self.n_chars)
        _y_logits = y_logits.reshape(bs * self.seq_len, self.n_chars)
        loss = torch.nn.functional.cross_entropy(_y_logits, _y, reduction='sum') / bs
        
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
        y = torch.argmax(y, dim=-1) # (bs, 2)
        y = y.detach().cpu().numpy() # (bs, 2)
        decoded = [ ''.join([ self.idx2char[idx] for idx in sample ]) for sample in y ]
        return decoded
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)