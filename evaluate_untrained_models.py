#!/usr/bin/env python3
"""
Script để đánh giá tất cả 60 model chưa qua training trên datasets SumMe và TVSum
Mục đích: So sánh hiệu năng random initialization vs trained models
"""

import os
import os.path as osp
import argparse
import h5py
import numpy as np
import torch
import json
from tabulate import tabulate
import time
from datetime import datetime

from models import DSN
from utils import read_json
import vsum_tools

# Cấu hình datasets
DATASETS = {
    'summe': {
        'h5_path': 'datasets/eccv16_dataset_summe_google_pool5.h5',
        'split_path': 'datasets/summe_splits.json',
        'metric': 'summe'
    },
    'tvsum': {
        'h5_path': 'datasets/eccv16_dataset_tvsum_google_pool5.h5', 
        'split_path': 'datasets/tvsum_splits.json',
        'metric': 'tvsum'
    }
}

# Cấu hình model architectures
MODEL_CONFIGS = {
    'DR-DSN': {'reward_type': 'dr', 'supervised': False, 'sup_only': False},
    'D-DSN': {'reward_type': 'd', 'supervised': False, 'sup_only': False},
    'D-DSN-nolambda': {'reward_type': 'd-nolambda', 'supervised': False, 'sup_only': False},
    'R-DSN': {'reward_type': 'r', 'supervised': False, 'sup_only': False},
    'DR-DSNsup': {'reward_type': 'dr', 'supervised': True, 'sup_only': False},
    'DSNsup': {'reward_type': 'dr', 'supervised': True, 'sup_only': True}
}

def create_untrained_model(model_type, input_dim=1024, hidden_dim=256, num_layers=1, rnn_cell='lstm'):
    """
    Tạo model chưa qua training với random weights.
    Đảm bảo cấu hình đúng với từng loại model (DR-DSN, DR-DSNsup, DSNsup, v.v.)
    
    Khác biệt giữa các model:
    - DR-DSN: Sử dụng cả diversity và representativeness trong reward
    - D-DSN: Chỉ sử dụng diversity trong reward
    - R-DSN: Chỉ sử dụng representativeness trong reward  
    - D-DSN-nolambda: Diversity nhưng không có lambda regularization
    - DR-DSNsup: DR-DSN có thêm supervised learning (kết hợp RL và SL)
    - DSNsup: Chỉ sử dụng supervised learning, không có RL
    
    Tham số:
    - model_type: Loại mô hình (DR-DSN, D-DSN, R-DSN, DR-DSNsup, ...)
    """
    config = MODEL_CONFIGS[model_type]
    
    # 1. Tạo model với random initialization
    model = DSN(in_dim=input_dim, hid_dim=hidden_dim, num_layers=num_layers, cell=rnn_cell)
    
    # 2. Thiết lập các thuộc tính phù hợp với loại model
    model.reward_type = config['reward_type']
    model.supervised = config['supervised']
    model.sup_only = config['sup_only']
    model.model_type = model_type  # Lưu thêm model type
    
    # 3. Log thông tin kiến trúc để debug
    model_str = (f"Model: {model_type} ("
                f"{'Supervised' if config['supervised'] else 'Unsupervised'}, "
                f"Reward: {config['reward_type']}"
                f"{', Supervised-only' if config['sup_only'] else ''})")
    print(f"  - Tạo untrained model: {model_str}")
    
    # Tạo một model description chi tiết
    if model_type == 'DR-DSN':
        model.description = "Diversity + Representativeness DSN (RL)"
    elif model_type == 'D-DSN':
        model.description = "Diversity-only DSN (RL)"
    elif model_type == 'R-DSN':
        model.description = "Representativeness-only DSN (RL)" 
    elif model_type == 'D-DSN-nolambda':
        model.description = "Diversity DSN without lambda (RL)"
    elif model_type == 'DR-DSNsup':
        model.description = "Diversity + Representativeness + Supervised DSN (RL+SL)"
    elif model_type == 'DSNsup':
        model.description = "Supervised-only DSN (SL)"
    
    # Không load weights, giữ nguyên random initialization
    return model

def evaluate_model(model, dataset, test_keys, metric_type, verbose=False):
    """Đánh giá model trên test set"""
    model.eval()
    eval_metric = 'avg' if metric_type == 'tvsum' else 'max'
    
    fms = []
    eval_arr = []
    
    with torch.no_grad():
        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            
            if torch.cuda.is_available():
                seq = seq.cuda()
            
            # Forward pass để lấy scores
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()
            
            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]
            
            # Tạo machine summary từ probability scores
            machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
            
            # Tính F-measure
            fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)
            
            eval_arr.append([key_idx, key, "{:.1%}".format(fm)])
            
            if verbose:
                print(f"  Video {key_idx+1}/{len(test_keys)}: {key} F1={fm:.1%}")
    
    mean_fm = np.mean(fms)
    return mean_fm, eval_arr

def evaluate_all_untrained_models(output_file='untrained_model_evaluation.json', verbose=False, seed=42):
    """Đánh giá tất cả 60 model combinations chưa qua training"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "ĐÁNH GIÁ 60 MODEL CHƯA QUA TRAINING")
    print("=" * 80)
    print(f"🕒 Bắt đầu lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Khởi tạo thời gian bắt đầu để tính toán tiến độ
    start_time_global = time.time()
    
    # Thiết lập seed cho reproducibility nếu cần
    if seed is not None:
        print(f"🔢 Sử dụng random seed: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
    
    results = {}
    summary_table = []
    
    # Thiết lập GPU nếu có
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        print(f"🖥️  Sử dụng GPU: {device_name} ({memory:.1f} GB)")
    else:
        print("💻 Sử dụng CPU để đánh giá")
    print()
    
    total_evaluations = len(MODEL_CONFIGS) * len(DATASETS) * 5  # 6 models × 2 datasets × 5 splits = 60
    current_eval = 0
    
    for model_type in MODEL_CONFIGS.keys():
        results[model_type] = {}
        
        for dataset_name in DATASETS.keys():
            dataset_config = DATASETS[dataset_name]
            results[model_type][dataset_name] = {}
            
            # Load dataset
            print(f"Loading dataset: {dataset_config['h5_path']}")
            dataset = h5py.File(dataset_config['h5_path'], 'r')
            splits = read_json(dataset_config['split_path'])
            
            split_scores = []
            
            for split_id in range(5):  # 5-fold cross validation
                current_eval += 1
                progress = (current_eval / total_evaluations) * 100
                elapsed_time = time.time() - start_time_global if 'start_time_global' in locals() else 0
                
                # Ước tính thời gian còn lại
                time_per_eval = elapsed_time / current_eval if current_eval > 0 else 0
                remaining_evals = total_evaluations - current_eval
                est_remaining = time_per_eval * remaining_evals if current_eval > 0 else "Đang tính..."
                
                # Định dạng thời gian còn lại
                if isinstance(est_remaining, str):
                    time_str = est_remaining
                else:
                    mins_remaining = int(est_remaining // 60)
                    secs_remaining = int(est_remaining % 60)
                    time_str = f"{mins_remaining}m {secs_remaining}s"
                
                print(f"[{progress:5.1f}%] [{current_eval:2d}/{total_evaluations}] "
                      f"Đánh giá {model_type:>10} trên {dataset_name.upper():>6} Split {split_id} "
                      f"(Còn lại: {time_str})")
                
                # Tạo model chưa qua training với kiến trúc phù hợp
                model = create_untrained_model(model_type)
                
                # Xác thực loại model đã được cấu hình đúng
                config = MODEL_CONFIGS[model_type]
                assert hasattr(model, 'reward_type'), f"Model không có thuộc tính reward_type"
                assert model.reward_type == config['reward_type'], f"reward_type không khớp: {model.reward_type} vs {config['reward_type']}"
                
                if use_gpu:
                    model = model.cuda()
                
                # Lấy test keys cho split này
                split = splits[split_id]
                test_keys = split['test_keys']
                
                # Đánh giá model
                start_time = time.time()
                mean_fm, eval_details = evaluate_model(
                    model, dataset, test_keys, 
                    dataset_config['metric'], verbose=verbose
                )
                eval_time = time.time() - start_time
                
                split_scores.append(mean_fm)
                results[model_type][dataset_name][f'split_{split_id}'] = {
                    'f1_score': mean_fm,
                    'eval_time': eval_time,
                    'details': eval_details
                }
                
                print(f"  ✓ F1-Score: {mean_fm:.1%} (thời gian: {eval_time:.1f}s)")
                
                # Cleanup GPU memory
                del model
                if use_gpu:
                    torch.cuda.empty_cache()
            
            # Tính trung bình cho dataset này
            mean_score = np.mean(split_scores)
            std_score = np.std(split_scores)
            
            results[model_type][dataset_name]['average'] = {
                'mean': mean_score,
                'std': std_score,
                'splits': split_scores
            }
            
            # Thêm vào bảng tổng kết
            splits_str = ", ".join([f"S{i}:{score:.1%}" for i, score in enumerate(split_scores)])
            summary_table.append([
                model_type,
                dataset_name.upper(),
                f"{mean_score:.1%} ± {std_score:.1%}",
                f"[{splits_str}]"
            ])
            
            dataset.close()
            print()
    
    # In kết quả tổng kết
    print("=" * 120)
    print("BẢNG TỔNG KẾT KẾT QUẢ ĐÁNH GIÁ MODEL CHƯA QUA TRAINING")
    print("=" * 120)
    
    headers = ["Model Type", "Dataset", "Average F1 ± Std", "Split Details"]
    print(tabulate(summary_table, headers=headers, tablefmt="grid"))
    
    # So sánh với trained models (nếu có)
    print("\n" + "=" * 80)
    print("SO SÁNH VỚI TRAINED MODELS")
    print("=" * 80)
    
    trained_scores = {
        'DR-DSN': {'summe': 0.399, 'tvsum': 0.564},
        'D-DSN': {'summe': 0.405, 'tvsum': 0.557},
        'R-DSN': {'summe': 0.388, 'tvsum': 0.566},
        'DR-DSNsup': {'summe': 0.419, 'tvsum': 0.567},
        'DSNsup': {'summe': 0.392, 'tvsum': 0.523}
    }
    
    comparison_table = []
    for model_type in MODEL_CONFIGS.keys():
        for dataset_name in DATASETS.keys():
            if model_type in trained_scores and dataset_name in trained_scores[model_type]:
                untrained_score = results[model_type][dataset_name]['average']['mean']
                trained_score = trained_scores[model_type][dataset_name]
                improvement = trained_score - untrained_score
                improvement_pct = (improvement / untrained_score) * 100 if untrained_score > 0 else 0
                
                comparison_table.append([
                    model_type,
                    dataset_name.upper(),
                    f"{untrained_score:.1%}",
                    f"{trained_score:.1%}",
                    f"+{improvement:.1%}",
                    f"+{improvement_pct:.1f}%"
                ])
    
    comp_headers = ["Model", "Dataset", "Untrained F1", "Trained F1", "Improvement", "Relative %"]
    print(tabulate(comparison_table, headers=comp_headers, tablefmt="grid"))
    
    # Lưu kết quả
    results['evaluation_info'] = {
        'timestamp': datetime.now().isoformat(),
        'total_models_evaluated': total_evaluations,
        'gpu_used': use_gpu,
        'random_seed': seed,
        'model_configs': {k: {'reward': v['reward_type'], 
                             'supervised': v['supervised'], 
                             'sup_only': v['sup_only']} for k, v in MODEL_CONFIGS.items()},
        'summary': summary_table,
        'comparison': comparison_table
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nKết quả đã được lưu vào: {output_file}")
    print(f"Hoàn thành lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Đánh giá tất cả model chưa qua training")
    parser.add_argument('--output', type=str, default='untrained_model_evaluation.json',
                       help="File output để lưu kết quả (default: untrained_model_evaluation.json)")
    parser.add_argument('--verbose', action='store_true',
                       help="Hiển thị chi tiết từng video")
    parser.add_argument('--seed', type=int, default=42,
                      help="Random seed để khởi tạo model reproducible (default: 42)")
    parser.add_argument('--no-seed', action='store_true',
                      help="Không thiết lập random seed (kết quả sẽ khác mỗi lần)")
    parser.add_argument('--no-gpu', action='store_true',
                      help="Không sử dụng GPU ngay cả khi khả dụng")
    
    args = parser.parse_args()
    
    # Kiểm tra datasets có tồn tại không
    for dataset_name, config in DATASETS.items():
        if not osp.exists(config['h5_path']):
            print(f"ERROR: Dataset {config['h5_path']} không tồn tại!")
            return
        if not osp.exists(config['split_path']):
            print(f"ERROR: Split file {config['split_path']} không tồn tại!")
            return
    
    # Vô hiệu hóa GPU nếu yêu cầu
    if args.no_gpu:
        print("Cờ --no-gpu được đặt: Chạy trên CPU thay vì GPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Vô hiệu hóa CUDA
    
    # Chạy đánh giá
    seed = None if args.no_seed else args.seed
    results = evaluate_all_untrained_models(args.output, args.verbose, seed=seed)

if __name__ == '__main__':
    main()
