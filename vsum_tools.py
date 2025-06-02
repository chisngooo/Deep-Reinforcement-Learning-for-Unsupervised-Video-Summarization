import numpy as np
from knapsack import knapsack_dp
import math
from scipy.spatial.distance import pdist, squareform

# Optional Ruptures import
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("Warning: ruptures library not available. Using traditional KTS only.")

def calc_kernel(feat, mode='cosine'):
    """Calculate kernel matrix for KTS (Kernel Temporal Segmentation).
    
    Improved version with better numerical stability and error handling.
    """
    n_frames = feat.shape[0]
    
    # Handle edge cases
    if n_frames <= 1:
        return np.ones((n_frames, n_frames), dtype=np.float32)
    
    # Check for zero or very small feature vectors
    feat_norms = np.linalg.norm(feat, axis=1)
    if np.any(feat_norms < 1e-8):
        print(f"Warning: Found {np.sum(feat_norms < 1e-8)} near-zero feature vectors")
        # Add small noise to zero vectors
        zero_mask = feat_norms < 1e-8
        feat[zero_mask] += np.random.normal(0, 1e-6, feat[zero_mask].shape)
    
    if mode == 'cosine':
        # Normalize features to unit vectors with better numerical stability
        feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
        # Compute cosine similarity matrix
        kernel = np.dot(feat_norm, feat_norm.T)
        # Ensure values are in [0,1] range (cosine similarity is in [-1,1])
        kernel = np.clip((kernel + 1) / 2.0, 0.0, 1.0)
        
    elif mode == 'euclidean':
        # Compute pairwise Euclidean distances
        distances = squareform(pdist(feat, 'euclidean'))
        # Convert to similarity with adaptive sigma
        sigma = np.median(distances[distances > 0])  # Use median instead of mean
        if sigma < 1e-8:
            sigma = 1.0  # Fallback value
        kernel = np.exp(-distances / sigma)
        
    elif mode == 'rbf':
        # RBF (Gaussian) kernel with improved gamma selection
        distances = squareform(pdist(feat, 'sqeuclidean'))
        # Use median heuristic for gamma
        gamma = 1.0 / (2 * np.median(distances[distances > 0]) + 1e-8)
        kernel = np.exp(-gamma * distances)
        
    else:
        raise ValueError(f"Unknown kernel mode: {mode}")
    
    # Ensure diagonal is 1 (self-similarity)
    np.fill_diagonal(kernel, 1.0)
    
    # Final clipping and type conversion
    kernel = np.clip(kernel, 0.0, 1.0)
    
    # Debug info
    print(f"DEBUG Kernel: shape={kernel.shape}, min/max={kernel.min():.4f}/{kernel.max():.4f}")
    print(f"DEBUG Kernel: mean={kernel.mean():.4f}, std={kernel.std():.4f}")
    
    return kernel.astype(np.float32)


def kts_with_ruptures(features, n_bkps=10, model="rbf"):
    """
    Thay thế KTS bằng thư viện ruptures
    
    Parameters:
    - features: (T, feature_dim) feature vectors
    - n_bkps: số breakpoints tối đa
    - model: 'rbf', 'cosine', 'linear', 'l1', 'l2'
    """
    
    if not RUPTURES_AVAILABLE:
        print("Warning: ruptures not available, falling back to traditional KTS")
        return None
    
    # 1. Sử dụng Kernel-based change point detection (tương tự KTS)
    if model == "rbf":
        # Tương đương với KTS kernel
        algo = rpt.KernelCPD(kernel="rbf", min_size=5).fit(features)
    elif model == "cosine":
        # Cosine similarity kernel
        algo = rpt.KernelCPD(kernel="cosine", min_size=5).fit(features)  
    else:
        # Các model khác
        algo = rpt.Pelt(model=model, min_size=5).fit(features)
    
    # 2. Detect change points
    try:
        change_points = algo.predict(n_bkps=n_bkps)
        # Remove the last point (end of signal)
        if change_points[-1] == len(features):
            change_points = change_points[:-1]
    except:
        # Fallback: uniform segmentation
        segment_length = len(features) // (n_bkps + 1)
        change_points = [i * segment_length for i in range(1, n_bkps + 1)]
        change_points = [cp for cp in change_points if cp < len(features)]
    
    return np.array(change_points)


def smart_kts(features, max_ncp=100, method="auto"):
    """
    Smart KTS wrapper that automatically selects the best method
    
    Parameters:
    - features: (T, feature_dim) feature vectors
    - max_ncp: maximum number of change points
    - method: 'auto', 'ruptures', 'traditional'
    """
    
    if method == "auto":
        # Choose best available method
        if RUPTURES_AVAILABLE and features.shape[0] > 50:
            method = "ruptures"
        else:
            method = "traditional"
    
    if method == "ruptures" and RUPTURES_AVAILABLE:
        print("Using Ruptures-based KTS")
        change_points = kts_with_ruptures(features, n_bkps=max_ncp, model="rbf")
        if change_points is not None:
            return change_points
        print("Ruptures failed, falling back to traditional KTS")
    
    # Traditional KTS
    print("Using traditional KTS")
    kernel = calc_kernel(features, mode='cosine')
    change_points, _ = cpd_auto(kernel, ncp=max_ncp, vmax=1e8)
    return change_points


def cpd_auto(kernel, ncp=1, vmax=1e8):
    """Automatic Change Point Detection using Kernel Temporal Segmentation (KTS).
    
    This implements the KTS algorithm from "Kernel Temporal Segmentation" (Potapov et al. 2014)
    for automatic detection of temporal boundaries (shot changes) in video sequences.
    
    The algorithm finds optimal segmentation points by maximizing within-segment similarity
    while minimizing between-segment similarity using dynamic programming.
    
    Parameters:
    -----------
    kernel : ndarray
        Symmetric kernel/similarity matrix of shape (n_frames, n_frames)
        Values should be in [0,1] where higher values indicate higher similarity
    ncp : int
        Maximum number of change points to detect (default: 1)
    vmax : float
        Maximum allowed variance for numerical stability (default: 1e8)
    
    Returns:
    --------
    change_points : ndarray
        Array of change point indices (frame numbers where shots change)
    scores : ndarray
        Quality scores for each segmentation (higher is better)
    """
    n_frames = kernel.shape[0]
    
    if ncp <= 0:
        return np.array([]), np.array([])
    
    if ncp >= n_frames - 1:
        return np.arange(1, n_frames), np.zeros(n_frames - 1)
    
    # Precompute cumulative kernel sums for efficiency
    # This allows O(1) computation of segment similarity scores
    kernel_cumsum = np.cumsum(np.cumsum(kernel, axis=0), axis=1)
    
    def get_segment_score(start, end):
        """Calculate within-segment similarity score for frames [start:end]."""
        if start >= end:
            return 0.0
        
        # Sum of kernel values within the segment
        if start == 0:
            segment_sum = kernel_cumsum[end-1, end-1]
        else:
            segment_sum = (kernel_cumsum[end-1, end-1] - 
                          kernel_cumsum[start-1, end-1] - 
                          kernel_cumsum[end-1, start-1] + 
                          kernel_cumsum[start-1, start-1])
        
        # Normalize by segment size squared
        segment_size = end - start
        if segment_size <= 0:
            return 0.0
        
        score = segment_sum / (segment_size * segment_size)
        return min(score, vmax)  # Cap at vmax for numerical stability
    
    # Dynamic programming to find optimal segmentation
    # dp[i][j] = best score for segmenting frames [0:i] with j change points
    dp = np.full((n_frames + 1, ncp + 1), -np.inf)
    parent = np.full((n_frames + 1, ncp + 1), -1, dtype=int)
    
    # Base case: no change points (single segment)
    for i in range(1, n_frames + 1):
        dp[i][0] = get_segment_score(0, i)
    
    # Fill DP table
    for num_cp in range(1, ncp + 1):
        for end in range(num_cp + 1, n_frames + 1):
            # Try all possible positions for the last change point
            for last_cp in range(num_cp, end):
                # Score = previous segments + current segment
                prev_score = dp[last_cp][num_cp - 1]
                curr_score = get_segment_score(last_cp, end)
                total_score = prev_score + curr_score
                
                if total_score > dp[end][num_cp]:
                    dp[end][num_cp] = total_score
                    parent[end][num_cp] = last_cp
    
    # Find optimal number of change points based on score improvement
    best_ncp = 0
    best_score = dp[n_frames][0]
    
    for num_cp in range(1, ncp + 1):
        if dp[n_frames][num_cp] > best_score:
            best_score = dp[n_frames][num_cp]
            best_ncp = num_cp
    
    # Backtrack to find change points
    change_points = []
    scores = []
    
    if best_ncp > 0:
        curr_pos = n_frames
        curr_ncp = best_ncp
        
        while curr_ncp > 0 and curr_pos > 0:
            cp = parent[curr_pos][curr_ncp]
            if cp > 0:
                change_points.append(cp)
                scores.append(dp[curr_pos][curr_ncp])
            curr_pos = cp
            curr_ncp -= 1
        
        change_points.reverse()
        scores.reverse()
    
    return np.array(change_points, dtype=int), np.array(scores)

def generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack'):
    """Tạo tóm tắt video dựa trên keyshot (vector nhị phân).
    
    Chức năng này được mô tả trong phần "Summary Generation" của paper:
    "For a test video, we apply a trained DSN to predict the frame-selection probabilities as importance scores. We compute shot-level scores by averaging frame-level scores within the same shot. For temporal segmentation, we use KTS proposed by (Potapov et al. 2014). To generate a summary, we select shots by maximizing the total scores while ensuring that the summary length does not exceed a limit, which is usually 15% of the video length. The maximization step is essentially the 0/1 Knapsack problem..."
    
    Quy trình tạo tóm tắt video như mô tả chi tiết trong paper:
    1. DSN dự đoán xác suất lựa chọn cho mỗi khung hình (điểm quan trọng) ypred
    2. Các khung hình được phân đoạn thành các shot bằng KTS (Kernel Temporal Segmentation)
    3. Điểm quan trọng của mỗi shot được tính bằng trung bình điểm của các khung hình trong shot
    4. Sử dụng thuật toán Knapsack để chọn các shot tối đa hóa tổng điểm 
       nhưng không vượt quá giới hạn độ dài (thường là 15% độ dài video)
    5. Tạo vector nhị phân đánh dấu các khung hình thuộc các shot được chọn
    
    Tham số:
    ---------------------------------------------
    - ypred: điểm quan trọng dự đoán cho mỗi khung hình được lấy mẫu, đến từ đầu ra của DSN.
    - cps: điểm thay đổi cảnh (change points), ma trận 2D, mỗi hàng chứa một đoạn video
           được tạo bởi thuật toán KTS (Kernel Temporal Segmentation).
    - n_frames: số khung hình ban đầu của video gốc.
    - nfps: số khung hình mỗi đoạn (shot) video.
    - positions: vị trí của các khung hình được lấy mẫu trong video gốc.
    - proportion: độ dài của tóm tắt video (so với độ dài video gốc), mặc định là 15% 
                 theo đúng như mô tả trong paper ("which is usually 15% of the video length").
    - method: phương pháp lựa chọn các đoạn video, ['knapsack', 'rank'].
      + 'knapsack': sử dụng thuật toán knapsack để tối đa hóa tổng điểm quan trọng trong giới hạn độ dài cho phép
                    (0/1 Knapsack problem như paper đề cập)
      + 'rank': chọn các đoạn có điểm cao nhất cho đến khi đạt đến giới hạn độ dài
    
    Trả về:
    ---------------------------------------------
    - summary: vector nhị phân chỉ định các khung hình được chọn vào tóm tắt (1) hoặc không (0)
               Đây là keyshot dạng nhị phân để đánh giá với ground truth summaries
    """
    n_segs = cps.shape[0]
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i == len(ypred):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = ypred[i]

    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx,0]), int(cps[seg_idx,1]+1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    limits = int(math.floor(n_frames * proportion))

    # Use only the standard algorithms from the paper
    if method == 'knapsack':
        picks = knapsack_dp(seg_score, nfps, n_segs, limits)
    elif method == 'rank':
        order = np.argsort(seg_score)[::-1].tolist()
        picks = []
        total_len = 0
        for i in order:
            if total_len + nfps[i] < limits:
                picks.append(i)
                total_len += nfps[i]
    else:
        # Default to knapsack if method not recognized
        picks = knapsack_dp(seg_score, nfps, n_segs, limits)

    summary = np.zeros((1), dtype=np.float32) # this element should be deleted
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            tmp = np.ones((nf), dtype=np.float32)
        else:
            tmp = np.zeros((nf), dtype=np.float32)
        summary = np.concatenate((summary, tmp))

    summary = np.delete(summary, 0) # delete the first element
    return summary

def evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
    """So sánh tóm tắt máy tạo ra với tóm tắt của người dùng (dựa trên keyshot).
    
    Chức năng này thực hiện đánh giá như mô tả trong phần "Evaluation metric" và "Evaluation settings" của paper:
    "For fair comparison with other approaches, we follow the commonly used protocol from (Zhang et al. 2016b) 
    to compute F-score as the metric to assess the similarity between automatic summaries and ground truth summaries."
    
    Trong phần "Evaluation settings" paper giải thích cụ thể:
    1. Đối với SumMe: sử dụng phương pháp max (lấy kết quả tốt nhất từ nhiều annotator)
    2. Đối với TVSum: sử dụng phương pháp avg (lấy trung bình kết quả từ nhiều annotator)
    
    Tham số:
    --------------------------------
    - machine_summary: vector nhị phân dạng ndarray chỉ định khung hình được chọn bởi máy.
                       Output từ hàm generate_summary ở trên.
    - user_summary: vector nhị phân dạng ndarray chỉ định khung hình được chọn bởi người dùng.
                   Có thể có nhiều tóm tắt của người dùng (n_users, n_frames) do mỗi người tạo
                   một tóm tắt khác nhau.
    - eval_metric: phương pháp đánh giá {'avg', 'max'} như trong phần "Evaluation settings"
      + 'avg': lấy trung bình kết quả so sánh với nhiều tóm tắt của người dùng (cho TVSum).
               "We use the average of importance scores instead" như paper đề cập.
      + 'max': lấy giá trị tối đa (tốt nhất) từ nhiều phép so sánh (cho SumMe).
               "We still use the 5FCV but we augment the training data in each fold" như paper đề cập.
    
    Trả về:
    --------------------------------
    - f_score: F-score đo lường mức độ chính xác của tóm tắt máy so với tóm tắt người dùng
               Đây là thước đo chính được báo cáo trong các bảng kết quả (Bảng 1-4) của paper
    - precision: độ chính xác (tỷ lệ khung hình đúng trong tóm tắt máy) - |M∩G|/|M|
    - recall: độ bao phủ (tỷ lệ khung hình quan trọng được giữ lại) - |M∩G|/|G|
               Trong đó M là tóm tắt máy và G là tóm tắt ground-truth (người dùng)
    """
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary.astype(np.float32)
    n_users,n_frames = user_summary.shape

    # Nhị phân hóa tóm tắt (đảm bảo các giá trị chỉ là 0 hoặc 1)
    machine_summary[machine_summary > 0] = 1  # Chuyển tất cả giá trị > 0 thành 1
    user_summary[user_summary > 0] = 1        # Chuyển tất cả giá trị > 0 thành 1

    # Điều chỉnh độ dài của tóm tắt máy để phù hợp với số khung hình
    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]  # Cắt bớt nếu dài hơn
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))  # Thêm các số 0 nếu ngắn hơn
        machine_summary = np.concatenate([machine_summary, zero_padding])

    f_scores = []  # Lưu trữ F-score của từng so sánh
    prec_arr = []  # Lưu trữ precision (độ chính xác)
    rec_arr = []   # Lưu trữ recall (độ bao phủ)

    # So sánh với từng tóm tắt của người dùng
    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx,:]  # Tóm tắt của người dùng thứ user_idx
        
        # Tính phần giao nhau giữa tóm tắt máy và tóm tắt người dùng
        # (số khung hình xuất hiện trong cả hai tóm tắt)
        overlap_duration = (machine_summary * gt_summary).sum()
        
        # Tính độ chính xác (precision) và độ bao phủ (recall) như được mô tả trong phần "Experimental Setup"
        # Trong đó đề cập đến việc sử dụng F-score trong phần "Evaluation metric"
        
        # Precision = |M∩G|/|M| 
        # Trong đó: M là tập hợp khung hình trong tóm tắt máy (machine_summary)
        # G là tập hợp khung hình trong tóm tắt người dùng (gt_summary)
        precision = overlap_duration / (machine_summary.sum() + 1e-8)  # Thêm 1e-8 để tránh chia cho 0
        
        # Recall = |M∩G|/|G| 
        recall = overlap_duration / (gt_summary.sum() + 1e-8)
        
        # Tính F-score (hài hòa giữa precision và recall)
        # F = 2·P·R/(P+R) (Công thức F1-score tiêu chuẩn)
        if precision == 0 and recall == 0:
            f_score = 0.
        else:
            f_score = (2 * precision * recall) / (precision + recall)
            
        # Lưu các giá trị đánh giá
        f_scores.append(f_score)
        prec_arr.append(precision)
        rec_arr.append(recall)

    if eval_metric == 'avg':
        final_f_score = np.mean(f_scores)
        final_prec = np.mean(prec_arr)
        final_rec = np.mean(rec_arr)
    elif eval_metric == 'max':
        final_f_score = np.max(f_scores)
        max_idx = np.argmax(f_scores)
        final_prec = prec_arr[max_idx]
        final_rec = rec_arr[max_idx]
    
    return final_f_score, final_prec, final_rec