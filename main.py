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
parser.add_argument('--ent-weight', type=float, default=0.1,
                    help="λ_entropy cho confidence penalty (Pereyra 2017)")

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
        np.random.shuffle(idxs) # Xáo trộn các chỉ số để tăng tính ngẫu nhiên khi huấn luyện

        for idx in idxs:
            key = train_keys[idx]
            seq = dataset[key]['features'][...] # Chuỗi đặc trưng của video, kích thước (seq_len, dim)
            seq = torch.from_numpy(seq).unsqueeze(0) # Thêm chiều batch, kích thước (1, seq_len, dim)
            if use_gpu: seq = seq.to('cuda')
            probs = model(seq) # Xác suất được mô hình dự đoán, kích thước (1, seq_len, 1)
            
            ######################################################
            # -------- supervised branch (only if enabled) -------
            ######################################################
            if args.supervised:
                # gtsummary đã khớp 2 fps (vector 0/1, shape (T,))
                labels = torch.from_numpy(dataset[key]['gtsummary'][...]).float()
                labels = labels.unsqueeze(0).to(probs.device)       # (1,T)

                # (a) cross-entropy with explicit reduction='mean'
                # Squeeze the last dimension of probs to match labels shape
                logits = probs.squeeze(-1)  # (1,T)
                ce = nn.functional.binary_cross_entropy(logits, labels, reduction='mean')

                # (b) confidence penalty  λ * H(p)
                # Use logits for consistency with BCE
                q = logits  # (1,T)
                entropy = -(q * q.clamp(1e-6).log() + 
                            (1-q) * (1-q).clamp(1e-6).log()).mean()
                sup_loss = ce + args.ent_weight * entropy           # L_sup

            # --- penalty giữ độ dài summary ---
            # L_percentage = ||1/T·∑_{t=1}^T p_t - ε||²
            # Trong đó: 
            # - ε là tỉ lệ khung hình cần chọn (mục tiêu là 0.5 tức 50%)
            # - p_t là xác suất lựa chọn khung hình t
            # - T là tổng số khung hình
            # Mục tiêu là để giữ độ dài tóm tắt khoảng 50% video gốc
            cost = args.beta * (probs.mean() - 0.5)**2        # L_percentage
            
            # Khởi tạo phân phối Bernoulli dựa trên xác suất dự đoán
            # Phân phối này được sử dụng để lấy mẫu các hành động nhị phân (0/1)
            # Đây là việc thực hiện công thức (2) trong paper: a_t ~ Bernoulli(p_t)
            m = Bernoulli(probs)
            epis_rewards = [] # Lưu phần thưởng cho mỗi episode
            
            # Thực hiện nhiều episode để ổn định việc học
            # Như paper đề cập trong phần "Training with Policy Gradient":
            # "Since Eq.(8) involves the expectation over high-dimensional action sequences, 
            # which is hard to compute directly, we approximate the gradient by running the agent for 
            # N episodes on the same video and then taking the average gradient"
            #
            # args.num_episode mặc định là 5 như paper đề cập trong phần "Optimization"
            for _ in range(args.num_episode):
                # Lấy mẫu hành động nhị phân từ phân phối Bernoulli (Công thức (2) trong paper)
                # a_t ~ Bernoulli(p_t)
                # Trong đó: 
                # - a_t là hành động (0/1) tại thời điểm t, cho biết khung hình t có được chọn hay không
                # - p_t là xác suất dự đoán bởi mạng DSN
                actions = m.sample()
                
                # Tính log xác suất của hành động đã lấy mẫu
                # Cần để tính gradient của log policy: ∇_θ log π_θ(a_t|h_t)
                # Đây là thành phần quan trọng trong công thức REINFORCE (8) và (9) trong paper
                log_probs = m.log_prob(actions).squeeze(-1)
                
                # Tính phần thưởng cho hành động đã lấy mẫu dựa trên reward_type
                # Phần thưởng này là R(S) trong công thức (6) của paper
                # Gọi với ignore_far_sim=True cho tất cả các trường hợp (λ=20 trong phần "Implementation details")
                # Hàm compute_reward sẽ tự động xử lý trường hợp d-nolambda bên trong
                reward = compute_reward(seq, actions, use_gpu=use_gpu, reward_type=args.reward_type)
                
                # Tính gradient của mất mát REINFORCE với baseline (Công thức (9) trong paper)
                # ∇_θJ(θ) ≈ 1/N·∑_{n=1}^N ∑_{t=1}^T R_n·∇_θ log π_θ(a_t|h_t)
                # Thực hiện: ∇_θ log π_θ(a_t|h_t)·(R_n-b) để giảm phương sai 
                # Trong đó:
                # - log π_θ(a_t|h_t): log xác suất của hành động được chọn
                # - R_n: phần thưởng nhận được từ episode n
                # - b: baseline (trung bình phần thưởng từ các episode trước) để giảm phương sai
                #
                # Như được đề cập trong phần "Training with Policy Gradient" của paper:
                # "A common countermeasure is to subtract the reward by a constant baseline b, 
                # so the gradient becomes ∇_θJ(θ) ≈ 1/N·∑_{n=1}^N ∑_{t=1}^T (R_n-b)·∇_θ log π_θ(a_t|h_t)"
                #
                # Dùng sum() thay vì mean() để tính tổng gradient, sẽ ổn định hơn cho triển khai PyTorch
                expected_reward = log_probs.sum() * (reward - baselines[key])
                
                # Cập nhật hàm mất mát (giảm giá trị âm của phần thưởng kỳ vọng)
                # Đây là phần L_RL của hàm mục tiêu tổng thể
                cost -= expected_reward          # L_RL
                epis_rewards.append(reward.item())
            
            # ============ thêm Cross-Entropy loss vào tổng loss ============
            if args.supervised:
                if args.sup_only:            # ----- DSNsup -----
                    # Chỉ sử dụng Cross-Entropy loss mà không có Reinforcement Learning
                    # Đây là biến thể giám sát hoàn toàn được đề cập trong phần "Extension to Supervised Learning"
                    # của paper, dùng để so sánh với phương pháp học tăng cường
                    cost = sup_loss + args.beta * (probs.mean() - 0.5)**2  # giữ CE+entropy+length
                else:                        # ----- DR-DSNsup ---
                    # Kết hợp cả Cross-Entropy loss và Reinforcement Learning
                    # Biến thể có giám sát được mô tả trong phần "Extension to Supervised Learning":
                    # "Given the keyframe indices for a video Y* = {y*_1,...,y*_|Y*|}, we use Maximum Likelihood Estimation (MLE)"
                    # "The objective is formulated as L_MLE = ∑_{t∈Y*} log p(t;θ)"
                    cost = cost + args.ce_weight * sup_loss

            # Thực hiện lan truyền ngược và cập nhật tham số mô hình
            optimizer.zero_grad() # Xóa gradient
            cost.backward() # Tính toán gradient
            # Giới hạn norm của gradient để ổn định huấn luyện (gradient clipping)
            # Kỹ thuật này giúp tránh gradient explosion, đặc biệt quan trọng khi huấn luyện RNN/LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) 
            optimizer.step() # Cập nhật tham số mô hình theo gradient đã tính toán
            
            # Cập nhật baseline bằng phương pháp trung bình di chuyển (moving average)
            # Trong paper, phần "Training with Policy Gradient" có đề cập:
            # "where b is simply computed as the moving average of rewards experienced so far for computational efficiency."
            #
            # b được tính bằng trung bình di chuyển của phần thưởng:
            # b ← γ·b + (1-γ)·R̄
            # Trong đó:
            # - γ = 0.9 là hệ số suy giảm (decay factor)
            # - b là baseline cũ
            # - R̄ là phần thưởng trung bình của các episode hiện tại
            #
            # Baseline này rất quan trọng để giảm phương sai trong quá trình huấn luyện REINFORCE
            # như được mô tả trong phần "Training with Policy Gradient" của paper
            baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards)
            reward_writers[key].append(np.mean(epis_rewards)) # Lưu phần thưởng trung bình cho mỗi video

        epoch_reward = np.mean([np.mean(reward_writers[key]) for key in reward_writers])
        print("epoch {}/{}\t reward {}\t".format(epoch+1, args.max_epoch, epoch_reward))
        
        # Cập nhật learning rate theo scheduler nếu có
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
    """
    Đánh giá mô hình trên tập dữ liệu kiểm thử
    Phương pháp đánh giá được mô tả trong phần "Summary Generation" và "Experiments"
    
    Tham số:
        model: Mô hình DSN đã được huấn luyện
        dataset: Tập dữ liệu chứa các video (SumMe hoặc TVSum)
        test_keys: Danh sách các khóa video dùng để kiểm thử
        use_gpu: Có sử dụng GPU không
    """
    print("==> Test")
    with torch.inference_mode():  # Sử dụng inference_mode của PyTorch để tăng hiệu suất
        model.eval()  # Chuyển mô hình sang chế độ đánh giá
        fms = []  # Lưu trữ các F-score
        eval_metric = 'avg' if args.metric == 'tvsum' else 'max'  # Lựa chọn phương pháp đánh giá dựa trên dataset

        if args.verbose: table = [["No.", "Video", "F-score"]]

        if args.save_results:
            h5_res = h5py.File(osp.join(args.save_dir, 'result.h5'), 'w')

        for key_idx, key in enumerate(test_keys):
            # Lấy đặc trưng của video từ dataset
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)  # Thêm chiều batch
            if use_gpu: seq = seq.to('cuda')
            
            # Dự đoán xác suất quan trọng cho mỗi khung hình
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()

            # Lấy thông tin về các điểm thay đổi (change points) và thông tin khác của video
            cps = dataset[key]['change_points'][...]  # Các điểm thay đổi cảnh
            num_frames = dataset[key]['n_frames'][()]  # Tổng số khung hình
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()  # Số khung hình mỗi đoạn
            positions = dataset[key]['picks'][...]  # Vị trí của các khung hình được lấy mẫu
            user_summary = dataset[key]['user_summary'][...]  # Tóm tắt do người dùng tạo

            # Tạo tóm tắt video dựa trên xác suất dự đoán và thuật toán knapsack
            # (giải quyết bài toán lựa chọn các đoạn video quan trọng nhất với ràng buộc về độ dài)
            machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
            
            # Đánh giá tóm tắt tự động với tóm tắt của người dùng
            # fm: F-score đo lường mức độ trùng khớp giữa tóm tắt máy và người dùng
            fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            if args.verbose:
                table.append([key_idx+1, key, "{:.1%}".format(fm)])

            if args.save_results:
                # Lưu kết quả vào file h5
                h5_res.create_dataset(key + '/score', data=probs)  # Điểm quan trọng dự đoán
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)  # Tóm tắt tự động
                h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])  # Điểm ground truth
                h5_res.create_dataset(key + '/fm', data=fm)  # F-score

    if args.verbose:
        print(tabulate(table))

    if args.save_results: h5_res.close()

    mean_fm = np.mean(fms)
    print("Average F-score {:.1%}".format(mean_fm))

    return mean_fm

if __name__ == '__main__':
    main()
