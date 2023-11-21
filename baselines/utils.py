import torch
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel, AdamW
from transformers import BertTokenizer, BertModel
from transformers.optimization import get_cosine_schedule_with_warmup
import json
import random
import numpy as np
from sklearn import metrics


def str2bool(str):
    if str == 'true':
        return True
    elif str == 'false':
        return False
    else:
        raise 'passed wrong just_test argument'

def load_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_tokenizer(model_path):
    try:
        #tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        return tokenizer
    except:
        raise Error('wrong model_path passed to initialize tokenizer')

def init_model(model_path):
    try:
        #model = AutoModel.from_pretrained(model_path)
        model = BertModel.from_pretrained(model_path)
        return model
    except:
        raise Error('wrong model_path passed to initialize model')

def init_optimizer(type, model, lr, weight_decay):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_params = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if type == 'adam':
        optimizer = Adam(optimizer_grouped_params, lr=lr)
    elif type == 'adamw':
        optimizer = AdamW(optimizer_grouped_params, lr=lr)
    else:
        raise Error('wrong optimizer type passed to initialize optimizer')
    return optimizer

def init_scheduler(type, optimizer, t_total, warmup_ratio):
    warmup_steps = int(t_total * warmup_ratio)

    if type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                            num_warmup_steps=warmup_steps, num_training_steps=t_total)
    else:
        raise Error('wrong scheduler type passed to initialize scheduler')
    return scheduler


def compute_metrics(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.tolist()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.tolist()
        
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    return accuracy, precision, recall, f1