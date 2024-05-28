import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Training script')

parser.add_argument('-t', '--type', type=int, required=True, 
                    help='Train type with configs 0 - Bert Base, 1 - DPR + BERT + GATConv, 2 - DPR + BERT + SageConv')
args = parser.parse_args()

def freeze_all_weights(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.bert.embeddings.parameters():
        param.requires_grad = True


def unfreeze_layers_by_block(model, block_num):
    for layer in range(block_num):
        for param in model.encoder.layer[layer].parameters():
            param.requires_grad = False
    
    for layer in range(block_num, model.config.num_hidden_layers):
        for param in model.encoder.layer[layer].parameters():
            param.requires_grad = True

    for param in model.pooler.parameters():
        param.requires_grad = True

def train(train_type):
    raw_datasets = load_dataset("imdb")
    tokenized_datasets = None
    if train_type > 0:
        from preprocessor import GraphPreprocessor
        preprocessor = GraphPreprocessor('bert-base-cased')
        tokenized_datasets = raw_datasets.map(lambda x: preprocessor.preprocess(x["text"]), batched=False)

    else:
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
        tokenized_datasets = raw_datasets.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched=False)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    if train_type == 1:
        from graph_tops import GAT
        model.bert.embeddings = GAT(768, 768)
    elif train_type == 2:
        from graph_tops import SageGAT
        model.bert.embeddings = SageGAT(768, 768)

    if train_type > 0:
        freeze_all_weights(model)

    from torch.utils.data import DataLoader
    import transformers
    batch_size = 1
    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=batch_size, collate_fn=transformers.default_data_collator)
    eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=batch_size, collate_fn=transformers.default_data_collator)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 5 if train_type == 0 else 7
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    _=model.to(device)

    progress_bar = tqdm(range(15468))

    criterion = nn.CrossEntropyLoss()
    accumulation_steps = 8
    model.train()
    for epoch in range(num_epochs):
        accumulated_steps = 0
        for batch in train_dataloader:
            predicted = None
            if train_type == 0:
                predicted = model(input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        token_type_ids=batch['token_type_ids'].to(device))
            else:
                predicted = model(input_ids = (batch['x'].to(device), batch['edges'].to(device),),
                                  attention_mask=batch['attention_mask'].to(device),
                        token_type_ids=batch['token_type_ids'].to(device))
                
            output = torch.softmax(predicted.logits, dim=1)
            loss = criterion(output, batch['labels'].to(device)) / accumulation_steps
            loss.backward()
            if accumulated_steps % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            accumulated_steps += 1
            progress_bar.update(1)

        true_samples = 0
        for batch in eval_dataloader:
            with torch.no_grad():
                if train_type == 0:
                    predicted = model(input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        token_type_ids=batch['token_type_ids'].to(device))
                else:
                    predicted = model(input_ids = (batch['x'].to(device), batch['edges'].to(device),),
                                  attention_mask=batch['attention_mask'].to(device),
                        token_type_ids=batch['token_type_ids'].to(device))
                true_samples += torch.sum(torch.softmax(predicted.logits, dim=1).cpu().data.argmax(dim=1) == batch['labels']) 
                loss = criterion(output, batch['labels'].to(device))
        
        accuracy = true_samples / (batch_size * len(eval_dataloader))
        
        print(f'Valudation accuracy {accuracy}')
        if epoch == 1:
            unfreeze_layers_by_block(model, max(12, epoch * 6)


