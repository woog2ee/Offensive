import torch
import torch.nn as nn
from utils import (str2bool, load_data, init_seed,
                   init_tokenizer, init_model,
                   init_optimizer, init_scheduler)
from dataset import CustomDataset
from torch.utils.data import DataLoader
from model import CustomModel
from iterator import iteration
import os
import random
import argparse
import numpy as np


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # disabling parallelism to avoid deadlocks

    parser = argparse.ArgumentParser()    
    parser.add_argument('--data_path', type=str,
                        default='/HDD/seunguk/OffensiveDataset/')
    parser.add_argument('--train_fname', type=str,
                        default='dojudge_train_clean.json')
    parser.add_argument('--valid_fname', type=str,
                        default='dojudge_valid_clean.json')

    parser.add_argument('--model_path', type=str,
                        default='monologg/koelectra-base-v3-discriminator')
    parser.add_argument('--save_path', type=str,
                        default='/HDD/seunguk/OffensiveTrained/')
    parser.add_argument('--result_path', type=str,
                        default='/HDD/seunguk/OffensiveResults/')
    parser.add_argument('--save_fname', type=str)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--use_scheduler', type=str2bool, default='true')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--clip', type=int, default=1)

    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--dropout_ratio', type=float, default=0.1)
    parser.add_argument('--just_test', type=str2bool, default='false')
    args = parser.parse_args()
    

    # initialize seed
    init_seed(args.seed)

    # initialize tokenizer
    tokenizer = init_tokenizer(args.model_path)
    pad_num = tokenizer('[PAD]', add_special_tokens=False)['input_ids'][0]
    print(f'* tokenizer with {args.model_path} initialized')

    # build dataset & dataloader
    train_data = load_data(args.data_path+args.train_fname)
    valid_data = load_data(args.data_path+args.valid_fname)
    if args.just_test == True:
        train_data = train_data[:500]
        valid_data = valid_data[:50]

    train_dataset = CustomDataset(data=train_data, tokenizer=tokenizer, max_len=args.max_len)
    valid_dataset = CustomDataset(data=valid_data, tokenizer=tokenizer, max_len=args.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f'* train & valid dataloader initialized')

    # initialize model
    bert = init_model(args.model_path)
    model = CustomModel(bert=bert, num_classes=2, dropout_ratio=args.dropout_ratio,
                        pad_num=pad_num).to(args.device)
    print(f'* detection model with {args.model_path} initialized')

    # initialize optimizer & scheduler
    loss_fn = nn.CrossEntropyLoss().to(args.device)
    optimizer = init_optimizer(type='adamw', model=model,
                               lr=args.lr, weight_decay=args.weight_decay)
    if args.use_scheduler:
        scheduler = init_scheduler(type='cosine', optimizer=optimizer,
                                   t_total=len(train_loader) * args.epochs,
                                   warmup_ratio=args.warmup_ratio)
    else:
        scheduler = None
    print(f'* optimizer & scheduler initialized')

    # train
    print(f'\n* training starts...')
    all_train_loss, all_valid_loss = [], []
    for epoch in range(args.epochs):
        train_loss = iteration('train', args, model, train_loader, epoch,
                               loss_fn, optimizer, scheduler)
        valid_loss = iteration('valid', args, model, valid_loader, epoch,
                               loss_fn, None, None)
        print('\n')
        all_train_loss.append(train_loss)
        all_valid_loss.append(valid_loss)
    
    # save results
    f = open(args.result_path+args.save_fname+'.txt', 'w')
    f.write(f' train loss: {all_train_loss}\n')
    f.write(f' valid loss: {all_valid_loss}\n')
    f.write(f' best epoch: {all_valid_loss.index(min(all_valid_loss))+1}')
    print(f'* train loss: {all_train_loss}')
    print(f'* valid loss: {all_valid_loss}')
    print(f'* best epoch: {all_valid_loss.index(min(all_valid_loss))+1}')
