import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from SentimentDataset import SentimentDataset
from transformers import AutoTokenizer

class SentimentDataModule_all(pl.LightningDataModule):

  def __init__(self, dataset, batch_size:8, max_token_length: 512,  model_name='roberta-base'):
    super().__init__()
    self.dataset = dataset
    self.batch_size = batch_size
    self.max_token_length = max_token_length
    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

  def setup(self, stage = None):
    self.dataset = SentimentDataset(self.dataset, tokenizer=self.tokenizer, max_length=self.max_token_length)

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

  def dataloader(self, shuffle=False):
    return DataLoader(self.dataset, batch_size = self.batch_size, num_workers=4, shuffle=shuffle, collate_fn=self.collate_fn)
