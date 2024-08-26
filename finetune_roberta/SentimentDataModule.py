import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from SentimentDataset import SentimentDataset
from transformers import AutoTokenizer

class SentimentDataModule(pl.LightningDataModule):

  def __init__(self, train_dataset, val_dataset, batch_size:8, max_token_length: 512,  model_name='roberta-base'):
    super().__init__()
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.batch_size = batch_size
    self.max_token_length = max_token_length
    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

  def setup(self, stage = None):
    if stage in (None, "fit"):
      self.train_dataset = SentimentDataset(self.train_dataset, tokenizer=self.tokenizer, max_length=self.max_token_length)
      self.val_dataset = SentimentDataset(self.val_dataset, tokenizer=self.tokenizer, max_length=self.max_token_length)
    if stage == 'predict':
      self.val_dataset = SentimentDataset(self.val_dataset, tokenizer=self.tokenizer, max_length=self.max_token_length)

  def collate_fn(self, batch):
        #print(batch)
        # Extract the input_ids, attention_mask, and labels from each item in the batch
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Convert lists to tensors and pad them to the same length
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=4, shuffle=True, collate_fn=self.collate_fn)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)

  def predict_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)
