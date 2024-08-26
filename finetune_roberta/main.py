import os
import torch
import random
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from datasets import load_metric, load_dataset, DatasetDict
from sklearn.metrics import classification_report
import json
from datetime import datetime
import argparse
import glob
import csv

from Sentiment_model import SentimentModel
from SentimentDataModule import SentimentDataModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_run_name(base_name="Run"):
    #logdir = '/lustre/umt3/user/guhitj/Erdos_bootcamp/Deeplearning/Project/Results/NewRun/tb_logs/'
    logdir = '/lustre/umt3/user/guhitj/Erdos_bootcamp/Deeplearning/Project/Results/NewRun/checkpoints/'
    existing_runs = [d for d in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, d)) and d.startswith(base_name)]
    run_number = len(existing_runs) + 1
    run_name = f"{base_name}_{run_number}"
    return run_name

def get_best_checkpoint(base_name="Run"):
    checkpoint_paths = '/lustre/umt3/user/guhitj/Erdos_bootcamp/Deeplearning/Project/Results/NewRun/checkpoints/'
    
    # List all directories that start with the base_name (e.g., "Run_")
    run_dirs = [d for d in os.listdir(checkpoint_paths) if os.path.isdir(os.path.join(checkpoint_paths, d)) and d.startswith(base_name)]

    # Sort the directories by their run number, extracted from the directory name
    run_dirs.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
    
    if not run_dirs:
        raise ValueError("No runs found.")

    # Get the most recent run directory
    latest_run_dir = os.path.join(checkpoint_paths, run_dirs[0])

    # List checkpoint files in the most recent run directory
    checkpoint_files = [f for f in os.listdir(latest_run_dir) if os.path.isfile(os.path.join(latest_run_dir, f))]

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {latest_run_dir}.")
    
    # Parse the filenames to find the one with the lowest val_loss
    best_checkpoint = None
    best_val_loss = float('inf')
    
    for checkpoint in checkpoint_files:
        # Assume the filename format is 'epoch=XX-val_loss=XX.ckpt'
        parts = checkpoint.split('val_loss=')
        if len(parts) == 2:
            val_loss_str = parts[1].replace('.ckpt', '')
            try:
                val_loss = float(val_loss_str)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint = checkpoint
            except ValueError:
                # Handle case where the conversion to float fails
                continue

    if best_checkpoint is None:
        raise ValueError(f"No valid checkpoint files found in {latest_run_dir}.")
    
    best_checkpoint_path = os.path.join(latest_run_dir, best_checkpoint)

    return best_checkpoint_path

def prepare_data(seed=1122456, use_small_subset=False):
    dataset_openai = load_dataset('csv', data_files='news_openai_final.csv')

    # Split the dataset into train, validation, and test sets
    train_val_test_split = dataset_openai['train'].train_test_split(test_size=0.2, seed=seed)
    train_val_split = train_val_test_split['train'].train_test_split(test_size=0.25, seed=seed)

    dataset = DatasetDict({
        'train': train_val_split['train'].shuffle(seed=1122456),  # 60% of the original data
        'validation': train_val_split['test'].shuffle(seed=1122456),  # 20% of the original data
        'test': train_val_test_split['test'].shuffle(seed=1122456),  # 20% of the original data
    })

    if use_small_subset:
        # Randomly select a small subset of data
        small_train_subset = dataset["train"].shuffle(seed=1122456).select([i for i in list(range(1000))])
        small_val_subset = dataset["validation"].shuffle(seed=1122456).select([i for i in list(range(200))])
        small_test_subset = dataset["test"].shuffle(seed=1122456).select([i for i in list(range(200))])

        dataset = DatasetDict({
            'train': small_train_subset,
            'validation': small_val_subset,
            'test': small_test_subset
        })

    return dataset

def train_model(dataset, logger, checkpoint_dir_run, huggingface_dir, finetune_from_checkpoint=False): 
    data_module = SentimentDataModule(dataset['train'], dataset['validation'], 8,  512 )

    config = {
        'model_name': 'roberta-base',
        'n_labels': 3,
        'batch_size': 8,
        'max_token_length': 512,
        'lr': 1.5e-6,
        'warmup': 0.2, 
        'train_size': len(data_module.train_dataloader()),
        'weight_decay': 0.001,
        'n_epochs': 20 #25 #100
    }

    labels = dataset['train']['openai_sentiment']
    class_weights = compute_class_weight(class_weight='balanced',
                                               classes=np.unique(labels),
                                               y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    reordered_class_weights = class_weights[[0, 1, 2]]

    # datamodule
    data_module = SentimentDataModule(
        dataset['train'], 
        dataset['validation'],
        batch_size=config['batch_size'],  
        max_token_length=config['max_token_length'],
        model_name=config['model_name']
    )

    # for finetuning
    if finetune_from_checkpoint:
        # load the latest checkpoint file
        latest_checkpoint_path = get_best_checkpoint()
        print("best checkpoint", latest_checkpoint_path)
        model = SentimentModel.load_from_checkpoint(latest_checkpoint_path)
        config['lr'] = config['lr'] * 0.1  # Reduce learning rate by 10x for fine-tuning
        best_lr = config['lr']
        best_val_loss = float('inf')
    else:
        # model to train from scratch
        model = SentimentModel(config, reordered_class_weights)

    # Unfreeze all layers (both for new model and fine-tuning)
    for param in model.parameters():
        param.requires_grad = True

    if not os.path.exists(checkpoint_dir_run):
        os.makedirs(checkpoint_dir_run)
        print(f"Directory '{checkpoint_dir_run}' created.")
    else:
        print(f"Directory '{checkpoint_dir_run}' already exists.")


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir_run,
        filename='{epoch:02d}-{val_loss:.5f}',  # Filename includes epoch and validation loss
        save_top_k=-1,  # Save all checkpoints (best and epoch checkpoints)
        monitor='val_loss',  # Monitor validation loss to determine the best model
        mode='min',  # Save the model with the minimum validation loss
        save_weights_only=False,  # Save the entire model (not just weights)
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitors the validation loss
        patience=5,  # Number of epochs with no improvement after which training will be stopped
        mode='min'  # Stop when the loss stops decreasing
    )

    # trainer and fit
    trainer = pl.Trainer(
        logger = logger, 
        max_epochs=config['n_epochs'], 
        accelerator='gpu', 
        devices=1, 
        num_sanity_val_steps=50, #50
        log_every_n_steps=25, 
        callbacks=[checkpoint_callback, early_stopping_callback] 
        ) 
    
    trainer.fit(model, data_module)

    if not os.path.exists(huggingface_dir):
        os.makedirs(huggingface_dir)
        print(f"Directory '{huggingface_dir}' created.")
    else:
        print(f"Directory '{huggingface_dir}' already exists.")

    #model.save_pretrained(huggingface_dir)
    #tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    #save_model_and_tokenizer(model, tokenizer, huggingface_dir, config=config)

    if finetune_from_checkpoint:
        current_val_loss = trainer.callback_metrics['val_loss'].item()
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_lr = config['lr']
        best_checkpoint_path = get_best_checkpoint(base_name="Run")
        model = SentimentModel.load_from_checkpoint(best_checkpoint_path)
        model.save_pretrained(huggingface_dir)
        return model, data_module, best_lr, best_val_loss
    else:
        best_checkpoint_path = get_best_checkpoint(base_name="Run")
        model = SentimentModel.load_from_checkpoint(best_checkpoint_path)
        model.save_pretrained(huggingface_dir)
        return model, data_module, None, None  # No learning rate tracking during regular training

    #return model, data_module

def evaluate(model, data_module, timestamp, perfdir, full_run_name):

    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    val_loader = data_module.val_dataloader()

    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_labels = []

    label_map = {0: -1, 1: 0, 2: 1}

    with torch.no_grad():
        for batch in val_loader:
            #print(batch)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            _, logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Compute predictions
            predictions = torch.argmax(logits, dim=-1)

            predictions_mapped = torch.tensor([label_map[pred.item()] for pred in predictions]).to(device)
            labels_mapped = torch.tensor([label_map[label.item()] for label in labels]).to(device)

            correct_predictions += (predictions_mapped == labels_mapped).sum().item()
            total_predictions += labels_mapped.size(0)
            all_predictions.extend(predictions_mapped.cpu().numpy())
            all_labels.extend(labels_mapped.cpu().numpy())

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    accuracy_filename = f'accuracy_{timestamp}.txt'
    full_path = os.path.join(perfdir, full_run_name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Directory '{full_path}' created.")
    else:
        print(f"Directory '{full_path}' already exists.")

    accuracy_path = os.path.join(full_path, accuracy_filename)
    with open(accuracy_path, 'w') as f:
        f.write(f"Validation Accuracy manual calculation: {accuracy:.4f}\n")

    # Classification report
    report = classification_report(all_labels, all_predictions, target_names=['Class -1', 'Class 0', 'Class 1'])
    report_filename = f'report_{timestamp}.txt'
    report_path = os.path.join(full_path, report_filename)
    with open(report_path, 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Train a sentiment model.')
    parser.add_argument('--small', action='store_true', help='Use a small subset of the data')
    parser.add_argument('--finetune', action='store_true', help='Path to checkpoint file for fine-tuning')
    parser.add_argument('--name_suffix', type=str, default='', help='Additional suffix for the run name')
    args = parser.parse_args()

    pl.seed_everything(1122456, workers=True)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    logdir = '/lustre/umt3/user/guhitj/Erdos_bootcamp/Deeplearning/Project/Results/NewRun/tb_logs/'
    perfdir = '/lustre/umt3/user/guhitj/Erdos_bootcamp/Deeplearning/Project/Results/NewRun/performance/'

    # Get the run name
    run_name = get_run_name()

    # Combine run name with timestamp
    if args.name_suffix:
        full_run_name = f"{run_name}_{timestamp}_{args.name_suffix}"
    else:
        full_run_name = f"{run_name}_{timestamp}"

    # Set up the logger with the full run name
    logger = TensorBoardLogger(logdir, name=full_run_name)

    checkpoint_dir = '/lustre/umt3/user/guhitj/Erdos_bootcamp/Deeplearning/Project/Results/NewRun/checkpoints'
    checkpoint_dir_run = os.path.join(checkpoint_dir, full_run_name)
    huggingface_dir = os.path.join(perfdir, full_run_name, "huggingface_model")

    # Prepare the data
    dataset = prepare_data(use_small_subset=args.small)
    # Train the model
    #model, data_module = train_model(dataset, logger, checkpoint_dir_run, finetune_from_checkpoint=args.finetune)
    model, data_module, best_lr, best_val_loss = train_model(dataset, logger, checkpoint_dir_run, huggingface_dir, finetune_from_checkpoint=args.finetune) 

    if args.finetune:
        best_lr_path = os.path.join(perfdir, full_run_name, f"best_lr_{timestamp}.txt")
        with open(best_lr_path, 'w') as f:
            f.write(f"Best Learning Rate: {best_lr}\n")
            f.write(f"Corresponding Validation Loss: {best_val_loss:.4f}\n")

    evaluate(model, data_module, timestamp, perfdir, full_run_name)


if __name__ == "__main__":
    main()
