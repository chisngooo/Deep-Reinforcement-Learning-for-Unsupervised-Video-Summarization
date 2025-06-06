from __future__ import print_function
import os
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

from utils import Logger, read_json, write_json, save_checkpoint
from models import *
from rewards import compute_reward
import vsum_tools

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options
parser.add_argument('-d', '--dataset', type=str, required=True, help="path to h5 dataset (required)")
parser.add_argument('-s', '--split', type=str, required=True, help="path to split file (required)")
parser.add_argument('--split-id', type=int, default=0, help="split index (default: 0)")
parser.add_argument('-m', '--metric', type=str, required=True, choices=['tvsum', 'summe'],
                    help="evaluation metric ['tvsum', 'summe']")
# Model options
parser.add_argument('--input-dim', type=int, default=1024, help="input dimension (default: 1024)")
parser.add_argument('--hidden-dim', type=int, default=256, help="hidden unit dimension of DSN (default: 256)")
parser.add_argument('--num-layers', type=int, default=1, help="number of RNN layers (default: 1)")
parser.add_argument('--rnn-cell', type=str, default='lstm', help="RNN cell type (default: lstm)")
# Optimization options
parser.add_argument('--lr', type=float, default=1e-05, help="learning rate (default: 1e-05)")
parser.add_argument('--weight-decay', type=float, default=1e-05, help="weight decay rate (default: 1e-05)")
parser.add_argument('--max-epoch', type=int, default=60, help="maximum epoch for training (default: 60)")
parser.add_argument('--stepsize', type=int, default=30, help="how many steps to decay learning rate (default: 30)")
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay (default: 0.1)")
parser.add_argument('--num-episode', type=int, default=5, help="number of episodes (default: 5)")
parser.add_argument('--beta', type=float, default=0.01, help="weight for summary length penalty term (default: 0.01)")
parser.add_argument('--reward-type', type=str, default='dr', choices=['dr', 'd', 'r', 'd-nolambda'],
                    help="reward type: dr (diversity+representativeness), d (diversity), r (representativeness), d-nolambda (diversity without lambda) (default: dr)")

# Supervised learning options
parser.add_argument('--supervised', action='store_true',
                    help="thêm CE loss dùng gtsummary")
parser.add_argument('--sup-only', action='store_true',
                    help="chỉ CE (DSNsup); nếu bỏ, CE sẽ cộng với RL (DR-DSNsup)")
parser.add_argument('--ce-weight', type=float, default=1.0,
                    help="β₃ – trọng số CE so với RL")

# Misc
parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
parser.add_argument('--use-cpu', action='store_true', help="use cpu device")
parser.add_argument('--evaluate', action='store_true', help="whether to do evaluation only")
parser.add_argument('--save-dir', type=str, default='log', help="path to save output (default: 'log/')")
parser.add_argument('--resume', type=str, default='', help="path to resume file")
parser.add_argument('--verbose', action='store_true', help="whether to show detailed test results")
parser.add_argument('--save-results', action='store_true', help="whether to save output results")

args = parser.parse_args()

# Set random seeds for reproducibility
torch.manual_seed(args.seed)
import random
random.seed(args.seed)
import numpy as np
np.random.seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
if args.use_cpu: use_gpu = False

def main():
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Initialize dataset {}".format(args.dataset))
    dataset = h5py.File(args.dataset, 'r')
    num_videos = len(dataset.keys())
    splits = read_json(args.split)
    assert args.split_id < len(splits), "split_id (got {}) exceeds {}".format(args.split_id, len(splits))
    split = splits[args.split_id]
    train_keys = split['train_keys']
    test_keys = split['test_keys']
    
    if args.supervised:
        train_keys = [k for k in train_keys if 'ovp' not in k and 'youtube' not in k]
        print("Using supervised mode: train videos with labels =", len(train_keys))
        
    print("# total videos {}. # train videos {}. # test videos {}".format(num_videos, len(train_keys), len(test_keys)))

    print("Initialize model")
    model = DSN(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    start_epoch = 0
    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device('cuda' if use_gpu else 'cpu'))
        model.load_state_dict(checkpoint)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        evaluate(model, dataset, test_keys, use_gpu)
        return

    print("==> Start training")
    start_time = time.time()
    model.train()
    baselines = {key: 0. for key in train_keys} # baseline rewards for videos
    reward_writers = {key: [] for key in train_keys} # record reward changes for each video

    for epoch in range(start_epoch, args.max_epoch):
        idxs = np.arange(len(train_keys))
        np.random.shuffle(idxs)

        for idx in idxs:
            key = train_keys[idx]
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            if use_gpu: seq = seq.to('cuda')
            probs = model(seq)
            
            # Supervised learning branch (if enabled)
            if args.supervised:
                labels = torch.from_numpy(dataset[key]['gtsummary'][...]).float()
                labels = labels.unsqueeze(0).to(probs.device)
                
                # Cross-entropy loss (Equation 14 in paper)
                logits = probs.squeeze(-1)
                ce = nn.functional.binary_cross_entropy(logits, labels, reduction='mean')
                sup_loss = ce

            # Length penalty (Equation 11 in paper)
            cost = args.beta * (probs.mean() - 0.5)**2
            
            # Sample actions using Bernoulli distribution
            m = Bernoulli(probs)
            epis_rewards = []
            
            # Run multiple episodes (default: 5 as mentioned in paper)
            for _ in range(args.num_episode):
                actions = m.sample()
                log_probs = m.log_prob(actions).squeeze(-1)
                
                # Compute reward (Equation 6 in paper)
                reward = compute_reward(seq, actions, use_gpu=use_gpu, reward_type=args.reward_type)
                
                # REINFORCE gradient with baseline (Equation 9 in paper)
                expected_reward = log_probs.sum() * (reward - baselines[key])
                cost -= expected_reward
                epis_rewards.append(reward.item())
            
            # ============ Supervised Learning Extension (theo paper gốc) ============
            if args.supervised:
                if args.sup_only:            # ----- DSNsup -----
                    # Chỉ sử dụng Cross-Entropy loss theo paper gốc (Equation 14)
                    # L_MLE = Σ log p(t|θ) cho t ∈ Y*
                    cost = sup_loss + args.beta * (probs.mean() - 0.5)**2
                else:                        # ----- DR-DSNsup ---
                    # Kết hợp Reinforcement Learning và Supervised Learning
                    # Theo paper: "Extension to Supervised Learning"
                    cost = cost + args.ce_weight * sup_loss

            # Thực hiện backpropagation và cập nhật tham số
            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            # Cập nhật baseline (moving average của rewards)
            baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards)
            reward_writers[key].append(np.mean(epis_rewards))

        epoch_reward = np.mean([np.mean(reward_writers[key]) for key in reward_writers])
        print("epoch {}/{}\t reward {}\t".format(epoch+1, args.max_epoch, epoch_reward))
        
        if args.stepsize > 0:
            scheduler.step()

    write_json(reward_writers, osp.join(args.save_dir, 'rewards.json'))
    evaluate(model, dataset, test_keys, use_gpu)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    model_state_dict = model.module.state_dict() if use_gpu else model.state_dict()
    model_save_path = osp.join(args.save_dir, 'model_epoch' + str(args.max_epoch) + '.pth.tar')
    save_checkpoint(model_state_dict, model_save_path)
    print("Model saved to {}".format(model_save_path))

    dataset.close()

def evaluate(model, dataset, test_keys, use_gpu):
    print("==> Test")
    with torch.inference_mode():
        model.eval()
        fms = []
        eval_metric = 'avg' if args.metric == 'tvsum' else 'max'

        if args.verbose: table = [["No.", "Video", "F-score"]]

        if args.save_results:
            h5_res = h5py.File(osp.join(args.save_dir, 'result.h5'), 'w')

        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            if use_gpu: seq = seq.to('cuda')
            
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]

            machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
            
            fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            if args.verbose:
                table.append([key_idx+1, key, "{:.1%}".format(fm)])

            if args.save_results:
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
                h5_res.create_dataset(key + '/fm', data=fm)

    if args.verbose:
        print(tabulate(table))

    if args.save_results: h5_res.close()

    mean_fm = np.mean(fms)
    print("Average F-score {:.1%}".format(mean_fm))

    return mean_fm

if __name__ == '__main__':
    main()
