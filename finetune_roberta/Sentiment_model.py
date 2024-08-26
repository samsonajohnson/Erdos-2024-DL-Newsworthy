import torch
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
import torch.nn as nn
import math
import pytorch_lightning as pl
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F
from torch.optim import AdamW
import os
import json
import csv

class SentimentModel(pl.LightningModule):
  def __init__(self, config, class_weights=None):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict = True)

    # Hidden layer to process output from the pretrained model
    self.hidden = nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)

    # Classifier layer that outputs logits for 3 classes if n_labels=3
    self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])

    # Initialize classifier weights using Xavier uniform initialization
    torch.nn.init.xavier_uniform_(self.classifier.weight)

    # Use CrossEntropyLoss for multi-class classification
    self.loss_func = nn.CrossEntropyLoss(weight=class_weights)

    self.dropout = nn.Dropout()

  def forward(self, input_ids, attention_mask, labels=None):
    # Pass inputs through the pretrained model
    output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)

    # Pool the output (mean over the sequence length)
    pooled_output = torch.mean(output.last_hidden_state, 1)

    # final logits
    pooled_output = F.relu(self.hidden(self.dropout(pooled_output)))
    
    # Get logits from classifier layer (shape: [batch_size, n_labels])
    logits = self.classifier(self.dropout(pooled_output))
   
    # calculate loss
    loss = None
    if labels is not None:
        # Compute the loss between logits and true labels without reshaping
        loss = self.loss_func(logits, labels)

    return loss, logits

  def training_step(self, batch, batch_index):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']

    loss, logits = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    preds = torch.argmax(logits, dim=1)
    acc = (preds == labels).float().mean()
    #loss, outputs = self(**batch)
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar = True, logger=True, sync_dist=True)
    self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    return {"loss":loss, "predictions":logits, "labels": batch["labels"]}

  def validation_step(self, batch, batch_index):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']

    loss, logits = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    preds = torch.argmax(logits, dim=-1)
    acc = (preds == labels).float().mean()
    
    self.log("val_loss", loss, prog_bar = True, logger=True)
    self.log("val_acc", acc, prog_bar=True, logger=True)
    return {"loss": loss, "preds": preds, "labels": labels}
  
  def predict_step(self, batch, batch_index):
    _, outputs = self(**batch)
    return outputs

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
    total_steps = self.config['train_size']/self.config['batch_size']
    warmup_steps = math.floor(total_steps * self.config['warmup'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return [optimizer],[scheduler]

  def save_pretrained(self, save_directory):
        """Save the model, including the pretrained part and custom layers."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the underlying Hugging Face model
        self.pretrained_model.save_pretrained(save_directory)

        # Save the custom layers
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))

        # Optionally save the configuration if it exists
        if self.config:
            config_path = os.path.join(save_directory, 'config.json')
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=4)
        
        # Save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        tokenizer.save_pretrained(save_directory)

        print(f"Model and tokenizer saved to {save_directory}")
