"""Entry point."""

import argparse
import torch
import trainer
import multi_trainer
import utils
import os
import numpy as np
import random

def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    # register_data_args(parser)
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'derive'],
                        help='train: Training ENAS, derive: Deriving Architectures')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    parser.add_argument('--cuda_num', type=int, default=0, help="GPU number")
    parser.add_argument('--save_epoch', type=int, default=5)
    parser.add_argument('--max_save_num', type=int, default=5)

    # controller
    parser.add_argument('--layers_of_child_model', type=int, default=3)
    parser.add_argument('--shared_initial_step', type=int, default=0, help ='step for child exploring')  #20, give up
    parser.add_argument('--controller_max_step', type=int, default=5, help='step for controller parameters')  #50
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--derive_num_sample', type=int, default=200)
    parser.add_argument('--search_mode', type=str, choices=['nas', 'shared'], default='nas')
    parser.add_argument('--controller_mode', type=str, default='multi', choices=['multi'])
    parser.add_argument('--num_selectCom', type=int, default=1,
                        help="number of action components selected to change the orignal architecture")
    parser.add_argument('--controller_grad_clip', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])
    parser.add_argument('--entropy_coeff', type=float, default=1e-4)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--ema_baseline_decay', type=float, default=0.95)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--controller_optim', type=str, default='adam')
    parser.add_argument('--controller_lr', type=float, default=3.5e-4,
                        help="will be ignored if --controller_lr_cosine=True")
    parser.add_argument('--tanh_c', type=float, default=2.5)
    parser.add_argument('--softmax_temperature', type=float, default=5.0)



    # child model
    parser.add_argument("--dataset", type=str, default="PPI", required=False,
                        help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="number of exploring epochs in exploring child model")
    parser.add_argument("--retrain_epochs", type=int, default=5,
                        help="number of epochs if the child model is retrained")
    parser.add_argument('--noRetrain_epochs', type=int, default=5,
                        help="number of epochs if the child model is not retrained")
    parser.add_argument('--fromScratch_epochs', type=int, default=300,
                        help="number of training epochs for the best model trained from scratch")
    parser.add_argument("--penalty_oversize", type=float, default=-0.01,
                        help="the reward penalty if the sampled model is oversized")
    parser.add_argument('--topK_actions', type=int, default=5,
                        help="number of top K architectures")
    parser.add_argument("--multi_label", type=bool, default=True,
                        help="multi_label or single_label task")
    parser.add_argument("--residual", default=False, action="store_true",
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--param_file", type=str, default="cora_test.pkl",
                        help="learning rate")
    parser.add_argument("--optim_file", type=str, default="opt_cora_test.pkl",
                        help="optimizer save path")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--max_param', type=float, default=5E6)
    parser.add_argument('--batch_normal', type=bool, default=True)
    parser.add_argument('--child_model_grad_clip', type=float, default=2.0)
    args = parser.parse_args()

    return args


def build_args_for_ppi(args):
    # args = build_args()
    if args.layers_of_child_model < 3:
        args.layers_of_child_model = 3
    args.in_feats = 50
    args.num_class = 121
    args.in_drop = 0
    args.weight_decay = 0
    args.lr = 0.005
    args.param_file = "ppi.pkl"
    args.optim_file = "ppi_optim.pkl"
    return args

def build_args_for_cora(args):
    args.layers_of_child_model = 2
    args.in_feats = 1433
    args.num_class = 7
    args.in_drop = 0.6
    args.weight_decay = 5e-4
    args.lr = 0.005
    args.controller_lr = 3.5e-4
    args.controller_grad_clip = 0
    args.child_model_grad_clip = 0
    args.fromScratch_epochs = 2000 # 5000
    args.controller_optim = 'adam'
    args.batch_normal = False    # False
    args.param_file = "cora.pkl"
    args.optim_file = "cora_optim.pkl"
    return args

def build_args_for_citeseer(args):
    args.layers_of_child_model = 2
    args.in_feats = 3703
    args.num_class = 6
    args.in_drop = 0.6
    args.weight_decay = 5e-4
    args.lr = 0.005
    args.controller_lr = 3.5e-4
    args.controller_grad_clip = 0
    args.child_model_grad_clip = 0
    args.fromScratch_epochs = 2000 # 5000
    args.controller_optim = 'adam'
    args.batch_normal = False
    args.param_file = "citeseer.pkl"
    args.optim_file = "citeseer_optim.pkl"
    return args

def build_args_for_pubmed(args):
    args.layers_of_child_model = 2
    args.in_feats = 500
    args.num_class = 3
    args.in_drop = 0.6
    args.weight_decay = 1e-3
    args.lr = 0.01
    args.controller_lr = 3.5e-4
    args.controller_grad_clip = 0
    args.child_model_grad_clip = 0
    args.fromScratch_epochs = 2000 # 5000
    args.controller_optim = 'adam'
    args.batch_normal = False
    args.param_file = "pubmed.pkl"
    args.optim_file = "pubmed_optim.pkl"
    return args


def main(args):
    # set random seed
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False

    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_num)
        # os.environ['CUDA_VISIBLE_DEVICES'] = "2"

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    #setup the record path
    utils.makedirs(os.path.join('params', args.dataset, args.controller_mode))
    utils.makedirs(os.path.join('log', args.dataset, args.controller_mode))
    if args.residual:
        skip = 'skip'
    else:
        skip = 'noSkip'
    if args.search_mode == "nas":
        epoch = args.retrain_epochs
    else:
        epoch = args.noRetrain_epochs
    if args.controller_mode in ['single', 'random']:
        args.param_file = os.path.join('params', args.dataset, args.controller_mode,
                                       f'{args.search_mode}_{skip}_epoch{epoch}_{args.param_file}')
        args.optim_file = os.path.join('params', args.dataset, args.controller_mode,
                                       f'{args.search_mode}_{skip}_epoch{epoch}_{args.optim_file}')
        args.sample_record = os.path.join('log', args.dataset, args.controller_mode,
                                          f'{args.search_mode}_{skip}_epoch{epoch}_sample_record.txt')
        args.search_record = os.path.join('log', args.dataset, args.controller_mode,
                                          f'{args.search_mode}_{skip}_epoch{epoch}_search_record.txt')
        args.train_time = os.path.join('log', args.dataset, args.controller_mode,
                                          f'{args.search_mode}_{skip}_epoch{epoch}_train_time.txt')
    else:
        args.param_file = os.path.join('params', args.dataset, args.controller_mode,
                                       f'{args.search_mode}_comp{args.num_selectCom}_{skip}_'
                                       f'epoch{epoch}_{args.param_file}')
        args.optim_file = os.path.join('params', args.dataset, args.controller_mode,
                                       f'{args.search_mode}_comp{args.num_selectCom}_{skip}_'
                                       f'epoch{epoch}_{args.optim_file}')
        args.sample_record = os.path.join('log', args.dataset, args.controller_mode,
                                          f'{args.search_mode}_comp{args.num_selectCom}_{skip}_'
                                          f'epoch{epoch}_sample_record.txt')
        args.search_record = os.path.join('log', args.dataset, args.controller_mode,
                                          f'{args.search_mode}_comp{args.num_selectCom}_{skip}_'
                                          f'epoch{epoch}_search_record.txt')
        args.train_time = os.path.join('log', args.dataset, args.controller_mode,
                                          f'{args.search_mode}_comp{args.num_selectCom}_{skip}_'
                                          f'epoch{epoch}_train_time.txt')


    # create files to record the search process
    print('Remove existing log records of search and sample: ', args.search_record, args.sample_record)
    utils.remove_file(args.search_record)
    utils.remove_file(args.sample_record)
    utils.remove_file(args.train_time)
    print('Removing existing parameters of child models')
    utils.remove_file(args.param_file)


    if args.controller_mode == 'multi' and args.num_selectCom > 1:
        trnr = multi_trainer.Trainer(args)
    else:
        trnr = trainer.Trainer(args)


    print(args)
    trnr.train()


if __name__ == "__main__":
    args = build_args()
    if args.dataset == "PPI":
        args = build_args_for_ppi(args)
    elif args.dataset == "Cora":
        args = build_args_for_cora(args)
    elif args.dataset == "Citeseer":
        args = build_args_for_citeseer(args)
    elif args.dataset == "Pubmed":
        args = build_args_for_pubmed(args)
    main(args)
