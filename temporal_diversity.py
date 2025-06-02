import numpy as np
import math
from typing import List, Tuple, Optional

def calculate_temporal_diversity_penalty(selected_segments: List[int], 
                                       n_segs: int, 
                                       penalty_weight: float = 0.3) -> float:
    """
    Tính penalty dựa trên việc phân phối temporal không đều
    
    Args:
        selected_segments: Danh sách các segment được chọn
        n_segs: Tổng số segment
        penalty_weight: Trọng số của penalty (0-1)
    
    Returns:
        penalty: Giá trị penalty (càng cao càng không đều)
    """
    if len(selected_segments) <= 1:
        return 0.0
    
    # Tính khoảng cách giữa các segment liên tiếp
    sorted_segments = sorted(selected_segments)
    gaps = []
    
    for i in range(len(sorted_segments) - 1):
        gap = sorted_segments[i + 1] - sorted_segments[i]
        gaps.append(gap)
    
    # Thêm khoảng cách từ đầu video đến segment đầu tiên
    gaps.insert(0, sorted_segments[0])
    
    # Thêm khoảng cách từ segment cuối cùng đến cuối video
    gaps.append(n_segs - 1 - sorted_segments[-1])
    
    # Tính độ lệch chuẩn của các khoảng cách
    # Phân phối đều sẽ có độ lệch chuẩn thấp
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)
    
    # Normalize penalty theo số segment
    normalized_penalty = std_gap / (mean_gap + 1e-8)
    
    return penalty_weight * normalized_penalty

def temporal_knapsack_selection(seg_scores: List[float], 
                              nfps: List[int], 
                              n_segs: int, 
                              limits: int,
                              diversity_weight: float = 0.2) -> List[int]:
    """
    Cải tiến của thuật toán knapsack với ràng buộc temporal diversity
    
    Args:
        seg_scores: Điểm số của các segment
        nfps: Số frame của mỗi segment
        n_segs: Tổng số segment
        limits: Giới hạn tổng số frame
        diversity_weight: Trọng số cho temporal diversity (0-1)
    
    Returns:
        selected_segments: Danh sách các segment được chọn
    """
    from knapsack import knapsack_dp
    
    # Bước 1: Chạy knapsack chuẩn để có baseline
    baseline_picks = knapsack_dp(seg_scores, nfps, n_segs, limits)
    
    # Bước 2: Tính diversity penalty cho baseline
    baseline_penalty = calculate_temporal_diversity_penalty(baseline_picks, n_segs)
    
    # Bước 3: Điều chỉnh scores dựa trên temporal position
    adjusted_scores = []
    for i, score in enumerate(seg_scores):
        # Tính temporal position (0 = đầu video, 1 = cuối video)
        temporal_pos = i / max(n_segs - 1, 1)
        
        # Penalty cho các segment ở đầu và cuối video
        position_penalty = diversity_weight * (2 * abs(temporal_pos - 0.5))
        
        # Điều chỉnh score
        adjusted_score = score * (1 - position_penalty)
        adjusted_scores.append(adjusted_score)
    
    # Bước 4: Chạy knapsack với adjusted scores
    adjusted_picks = knapsack_dp(adjusted_scores, nfps, n_segs, limits)
    
    # Bước 5: So sánh và chọn kết quả tốt nhất
    adjusted_penalty = calculate_temporal_diversity_penalty(adjusted_picks, n_segs)
    
    # Tính tổng score cho cả hai phương án
    baseline_total_score = sum(seg_scores[i] for i in baseline_picks) - baseline_penalty
    adjusted_total_score = sum(seg_scores[i] for i in adjusted_picks) - adjusted_penalty
    
    if adjusted_total_score > baseline_total_score:
        return adjusted_picks
    else:
        return baseline_picks

def sliding_window_selection(seg_scores: List[float], 
                           nfps: List[int], 
                           n_segs: int, 
                           limits: int,
                           window_size: Optional[int] = None) -> List[int]:
    """
    Phương pháp sliding window để đảm bảo phân phối đều
    
    Args:
        seg_scores: Điểm số của các segment
        nfps: Số frame của mỗi segment
        n_segs: Tổng số segment
        limits: Giới hạn tổng số frame
        window_size: Kích thước cửa sổ (None = tự động)
    
    Returns:
        selected_segments: Danh sách các segment được chọn
    """
    if window_size is None:
        # Tự động tính window size để có khoảng 5-10 cửa sổ
        window_size = max(1, n_segs // 7)
    
    selected = []
    total_frames = 0
    
    for start in range(0, n_segs, window_size):
        end = min(start + window_size, n_segs)
        window_segments = list(range(start, end))
        
        # Trong mỗi cửa sổ, chọn segment có điểm cao nhất nếu còn budget
        window_scores = [(i, seg_scores[i]) for i in window_segments]
        window_scores.sort(key=lambda x: x[1], reverse=True)
        
        for seg_idx, score in window_scores:
            if total_frames + nfps[seg_idx] <= limits:
                selected.append(seg_idx)
                total_frames += nfps[seg_idx]
                break  # Chỉ chọn 1 segment per window
    
    return selected

def uniform_sampling_selection(seg_scores: List[float], 
                             nfps: List[int], 
                             n_segs: int, 
                             limits: int) -> List[int]:
    """
    Phương pháp uniform sampling để đảm bảo phân phối đều tuyệt đối
    
    Args:
        seg_scores: Điểm số của các segment
        nfps: Số frame của mỗi segment
        n_segs: Tổng số segment
        limits: Giới hạn tổng số frame
    
    Returns:
        selected_segments: Danh sách các segment được chọn
    """
    # Ước tính số segment có thể chọn
    avg_nfps = np.mean(nfps)
    est_num_segments = min(n_segs, int(limits / avg_nfps))
    
    if est_num_segments <= 0:
        return []
    
    # Tính interval để phân phối đều
    interval = n_segs / est_num_segments
    
    selected = []
    total_frames = 0
    
    for i in range(est_num_segments):
        # Tính vị trí target
        target_pos = i * interval
        
        # Tìm segment gần nhất với điểm số cao nhất trong khoảng ±interval/2
        search_start = max(0, int(target_pos - interval/2))
        search_end = min(n_segs, int(target_pos + interval/2))
        
        best_seg = None
        best_score = -1
        
        for seg_idx in range(search_start, search_end):
            if total_frames + nfps[seg_idx] <= limits and seg_scores[seg_idx] > best_score:
                best_seg = seg_idx
                best_score = seg_scores[seg_idx]
        
        if best_seg is not None:
            selected.append(best_seg)
            total_frames += nfps[best_seg]
    
    return selected

def hybrid_selection(seg_scores: List[float], 
                    nfps: List[int], 
                    n_segs: int, 
                    limits: int,
                    methods: List[str] = ['knapsack', 'temporal_knapsack', 'sliding_window', 'uniform']) -> List[int]:
    """
    Kết hợp nhiều phương pháp và chọn kết quả tốt nhất
    
    Args:
        seg_scores: Điểm số của các segment
        nfps: Số frame của mỗi segment
        n_segs: Tổng số segment
        limits: Giới hạn tổng số frame
        methods: Danh sách các phương pháp để thử
    
    Returns:
        selected_segments: Danh sách các segment được chọn tốt nhất
    """
    from knapsack import knapsack_dp
    
    results = []
    
    for method in methods:
        try:
            if method == 'knapsack':
                picks = knapsack_dp(seg_scores, nfps, n_segs, limits)
            elif method == 'temporal_knapsack':
                picks = temporal_knapsack_selection(seg_scores, nfps, n_segs, limits)
            elif method == 'sliding_window':
                picks = sliding_window_selection(seg_scores, nfps, n_segs, limits)
            elif method == 'uniform':
                picks = uniform_sampling_selection(seg_scores, nfps, n_segs, limits)
            else:
                continue
            
            # Tính điểm tổng hợp
            total_score = sum(seg_scores[i] for i in picks)
            diversity_penalty = calculate_temporal_diversity_penalty(picks, n_segs)
            combined_score = total_score - diversity_penalty
            
            results.append((method, picks, combined_score, total_score, diversity_penalty))
            
        except Exception as e:
            print(f"Error in method {method}: {e}")
            continue
    
    if not results:
        # Fallback to simple knapsack
        from knapsack import knapsack_dp
        return knapsack_dp(seg_scores, nfps, n_segs, limits)
    
    # Chọn kết quả tốt nhất dựa trên combined score
    best_result = max(results, key=lambda x: x[2])
    
    print(f"Selected method: {best_result[0]}")
    print(f"Total score: {best_result[3]:.3f}")
    print(f"Diversity penalty: {best_result[4]:.3f}")
    print(f"Combined score: {best_result[2]:.3f}")
    
    return best_result[1]

def analyze_temporal_distribution(selected_segments: List[int], n_segs: int) -> dict:
    """
    Phân tích phân phối temporal của các segment được chọn
    
    Args:
        selected_segments: Danh sách các segment được chọn
        n_segs: Tổng số segment
    
    Returns:
        analysis: Dictionary chứa các thống kê phân phối
    """
    if not selected_segments:
        return {"error": "No segments selected"}
    
    # Normalize positions to [0, 1]
    positions = [seg / max(n_segs - 1, 1) for seg in selected_segments]
    
    # Tính các thống kê
    analysis = {
        "num_selected": len(selected_segments),
        "coverage": len(selected_segments) / n_segs,
        "mean_position": np.mean(positions),
        "std_position": np.std(positions),
        "min_position": np.min(positions),
        "max_position": np.max(positions),
        "range": np.max(positions) - np.min(positions)
    }
    
    # Phân loại phân phối
    if analysis["std_position"] < 0.2:
        analysis["distribution_type"] = "concentrated"
    elif analysis["range"] < 0.6:
        analysis["distribution_type"] = "partial"
    else:
        analysis["distribution_type"] = "distributed"
    
    # Kiểm tra bias về đầu/cuối
    if analysis["mean_position"] < 0.3:
        analysis["bias"] = "beginning"
    elif analysis["mean_position"] > 0.7:
        analysis["bias"] = "end"
    else:
        analysis["bias"] = "balanced"
    
    return analysis
