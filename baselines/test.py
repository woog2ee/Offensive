import torch
from utils import load_data, init_tokenizer
from dataset import CustomDataset
from torch.utils.data import DataLoader
from iterator import iteration
import os
import argparse


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # disabling parallelism to avoid deadlocks

    parser = argparse.ArgumentParser()    
    parser.add_argument('--data_path', type=str,
                        default='/HDD/seunguk/OffensiveDataset/')
    parser.add_argument('--test_fname', type=str,
                        default='dojudge_test_clean.json')

    parser.add_argument('--model_path', type=str,
                        default='monologg/koelectra-base-v3-discriminator')
    parser.add_argument('--save_path', type=str,
                        default='/HDD/seunguk/OffensiveTrained/')
    parser.add_argument('--save_fname', type=str)
    parser.add_argument('--save_epoch', type=int)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=10)

    parser.add_argument('--max_len', type=int, default=64)
    args = parser.parse_args()


    # initialize tokenizer
    tokenizer = init_tokenizer(args.model_path)
    pad_num = tokenizer('[PAD]', add_special_tokens=False)['input_ids'][0]
    print(f'* tokenizer with {args.model_path} initialized')

    # build dataset & dataloader
    test_data = load_data(args.data_path+args.test_fname)
    test_dataset = CustomDataset(data=test_data, tokenizer=tokenizer, max_len=args.max_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f'* test dataloader from {args.data_path+args.test_fname} initialized')

    # load finetuned model
    save_path = args.save_path+args.save_fname+f'_{args.save_epoch}.pt'
    model = torch.load(save_path)
    print(f'* trained model {save_path} loaded')

    # test
    print(f'\n* testing starts...')
    metrics = iteration('test', args, model, test_loader, 0, None, None, None)
    print(f'* test accuracy: {metrics[0]}')
    print(f'* test precision: {metrics[1]}')
    print(f'* test recall: {metrics[2]}')
    print(f'* test f1-score: {metrics[3]}')