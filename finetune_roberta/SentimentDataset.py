import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        headline = str(item['Headline'])
        text = str(item['Text'])
        combined_text = headline + " " + text

        # Tokenize the text with truncation and padding
        encoding = self.tokenizer(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,  # Truncate to max_length if necessary
            padding='max_length',  # Pad sequences to max_length
            return_tensors='pt',
            return_attention_mask=True  # Generate attention mask
        )

        # Shift labels from -1, 0, 1 to 0, 1, 2
        label = torch.tensor(item['openai_sentiment'] + 1, dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].squeeze(dim=0),
            'attention_mask': encoding['attention_mask'].squeeze(dim=0),
            'labels': label
        }