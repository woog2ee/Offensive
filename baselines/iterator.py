import torch
import torch.nn as nn
from utils import compute_metrics
from tqdm import tqdm


def iteration(type, args, model, data_loader, epoch,
              loss_fn, optimizer, scheduler):

    tqdm_iter = tqdm(enumerate(data_loader),
                     desc='Epoch_%s:%d' % (type, epoch+1),
                     total=len(data_loader),
                     bar_format='{l_bar}{r_bar}')

    if type == 'train':
        epoch_loss = 0.0
        model.train()
    elif type == 'valid':
        epoch_loss = 0.0
        model.eval()
    elif type == 'test':
        all_accuracy, all_precision, all_recall, all_f1 = 0.0, 0.0, 0.0, 0.0
        model.eval()
    else:
        raise Error('wrong iteration type passed to train/evaluate/test the model')


    for idx, batch in tqdm_iter:
        if type == 'train':
            optimizer.zero_grad()

        batch = {k: v.to(args.device) for k, v in batch.items()}
        input_ids, label = batch['input_ids'], batch['label']
        # input_ids: [batch_size, max_len] [32, 64]
        # label: [batch_size] [32]

        out = model(input_ids)
        # out: [batch_size, 2] [32, 2]

        if type in ['train', 'valid']:
            loss = loss_fn(out, label)
            epoch_loss += loss.item()
    
        if type == 'train':
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            if scheduler != None: scheduler.step()

        if type in ['train', 'valid']:
            post_fix = {'epoch': epoch+1,
                        'batch': idx+1,
                        'loss': epoch_loss / (idx+1)}
            if (idx+1) % 50 == 0:
                tqdm_iter.write(str(post_fix))
        if type == 'test':
            print('\n')
            print(out)
            print('\n')

            print('out')
            out = torch.argmax(out, dim=1)
            print(out)

            print('label')
            print(label)
            exit()

            accuracy, precision, recall, f1 = compute_metrics(label, out)
            all_accuracy += accuracy
            all_precision += precision
            all_recall += recall
            all_f1 += f1


    # save model
    if type == 'train':
        save_path = args.save_path+args.save_fname+f'_{epoch+1}.pt'
        torch.save(model, save_path)
        print(f'* model saved at {save_path}')


    # return results
    if type in ['train', 'valid']:
        epoch_loss /= (idx+1)
        return epoch_loss
    elif type == 'test':
        all_accuracy /= (idx+1)
        all_precision /= (idx+1)
        all_recall /= (idx+1)
        all_f1 /= (idx+1)
        return [all_accuracy, all_precision, all_recall, all_f1]