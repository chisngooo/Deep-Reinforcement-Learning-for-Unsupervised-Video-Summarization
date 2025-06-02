import torch
import sys

def compute_reward(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False, reward_type='dr'):
    """
    Tính toán phần thưởng dựa trên tính đa dạng (diversity) và tính đại diện (representativeness)
    như mô tả trong phần "Diversity-Representativeness Reward Function"
    
    Theo paper, một tóm tắt chất lượng cao nên đồng thời đa dạng và có tính đại diện. 
    DR reward function đánh giá chất lượng tóm tắt bằng cách kết hợp hai phần thưởng:
    1. R_div: đo lường mức độ đa dạng giữa các khung hình được chọn (khác nhau như thế nào)
    2. R_rep: đo lường mức độ đại diện của các khung hình được chọn cho toàn bộ video
    
    Hàm này thực hiện tính toán:
    R(S) = R_div + R_rep  (Công thức (6) trong paper)
    
    Đây là phần trung tâm của khung học tăng cường được mô tả trong Hình 1, 
    nơi phần thưởng R(S) được tính toán dựa vào chất lượng tóm tắt và được 
    sử dụng để huấn luyện DSN thông qua gradient policy.
    
    Tham số:
        seq: chuỗi đặc trưng, kích thước (1, seq_len, dim)
        actions: chuỗi hành động nhị phân, kích thước (1, seq_len, 1)
        ignore_far_sim (bool): có bỏ qua sự tương đồng về mặt thời gian không (mặc định: True)
                              đặt d(x_i, x_j) = 1 nếu |i - j| > λ như đề cập trong phần đầu paper
        temp_dist_thre (int): ngưỡng λ để bỏ qua sự tương đồng về mặt thời gian (mặc định: 20)
                              Như mô tả trong phần "Implementation details": λ được đặt là 20
        use_gpu (bool): có sử dụng GPU không
        reward_type (str): loại phần thưởng sử dụng ('dr', 'd', 'r', 'd-nolambda')
                          'dr': sử dụng cả diversity và representativeness (DR-DSN)
                          'd': chỉ sử dụng diversity (D-DSN)
                          'r': chỉ sử dụng representativeness (R-DSN)
                          'd-nolambda': chỉ sử dụng diversity nhưng không bỏ qua sự tương đồng xa về thời gian
                          
                          Các biến thể này được đánh giá trong Bảng 1 và Bảng 2 của paper
    
    Trả về:
        reward: Phần thưởng tổng hợp dựa theo reward_type
    """
    # Tách gradient để tính toán phần thưởng
    _seq = seq.detach()
    _actions = actions.detach()
    
    # Thiết lập device dựa vào input sequence để đảm bảo nhất quán
    device = seq.device
    
    # Lấy vị trí của các khung hình được chọn (giá trị 1 trong actions)
    # Tương ứng với S = {i|a_i = 1, i = 1,2,...} trong paper
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    
    if num_picks == 0:
        # Trả về phần thưởng 0 nếu không có khung hình nào được chọn
        # Như đề cập trong paper: "We give zero reward to DSN when no frames are selected"
        # Đây là một phần trong việc huấn luyện DSN - không nên có trường hợp không chọn khung hình nào
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.to(device)
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # Tính toán phần thưởng về tính đa dạng (diversity reward - R_div)
    if num_picks == 1:
        # Nếu chỉ chọn 1 khung hình, không có sự đa dạng vì không có hai khung hình để so sánh
        # Theo công thức (3), không thể tính mức độ đa dạng với chỉ một khung hình
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.to(device)
    else:
        # Chuẩn hóa các vector đặc trưng để tính cosine similarity
        # Điều này chuẩn bị cho việc tính d(x_i, x_j) = 1 - cos(x_i, x_j)
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        
        # Tính ma trận khác biệt (Công thức (4) trong paper)
        # d(x_i, x_j) = 1 - (x_i·x_j)/(||x_i||·||x_j||)
        # Trong đó: 
        # - d(x_i, x_j) là độ khác biệt giữa khung i và j
        # - (x_i·x_j)/(||x_i||·||x_j||) là độ tương đồng cosine
        # Trực quan: giá trị càng gần 1 thì hai khung hình càng khác nhau
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())
        
        # Lấy ma trận con chỉ chứa các khung hình được chọn vào tóm tắt
        # Đây là ma trận d(x_i, x_j) với i,j thuộc S (các khung hình được chọn)
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
        
        if ignore_far_sim:
            # Trong paper có đề cập ở phần đầu: đặt d(x_i, x_j) = 1 nếu |i - j| > λ
            # Đây là một giả thuyết quan trọng: khung hình cách xa nhau về thời gian
            # nên được xem là hoàn toàn khác nhau để đảm bảo tính đa dạng của tóm tắt
            # và tránh tập trung quá nhiều vào một phần của video
            pick_mat = pick_idxs.unsqueeze(0).expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            
            # Đặt giá trị là 1 (khác biệt hoàn toàn) cho các cặp khung hình cách xa nhau
            # λ = temp_dist_thre (mặc định là 20 như trong phần "Implementation details")
            # Đây là thực hiện của quy tắc: d(x_i, x_j) = 1 nếu |i - j| > λ
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        
        # Tính phần thưởng đa dạng (Công thức (3) trong paper)
        # R_div = (∑_{i∈S} ∑_{j∈S,j≠i} d(x_i,x_j))/(|S|·(|S|-1))
        # Trong đó: 
        # - S = {i|a_i = 1, i = 1,2,...} là tập hợp các khung hình được chọn
        # - |S| là số lượng khung hình được chọn (num_picks)
        # - d(x_i,x_j) là độ khác biệt giữa hai khung hình i và j
        # 
        # Trực quan: R_div càng cao khi các khung hình được chọn càng khác nhau
        # Mục tiêu: Khuyến khích tóm tắt chứa các khung hình đa dạng, không bị trùng lặp
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))

    # Tính toán phần thưởng về tính đại diện (representativeness reward - R_rep)
    
    # Tính ma trận khoảng cách Euclidean bình phương giữa các vector đặc trưng
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i·x_j
    # dist_mat = kích thước (n, n) với n là số khung hình
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)  # ||x_i||^2
    dist_mat = dist_mat + dist_mat.t()  # ||x_i||^2 + ||x_j||^2
    dist_mat.addmm_(beta=1, mat1=_seq, mat2=_seq.t(), alpha=-2)  # trừ 2*x_i·x_j
    
    # Lấy khoảng cách từ mỗi khung hình trong video đến các khung hình được chọn vào tóm tắt
    # dist_mat[:,pick_idxs]: khoảng cách từ mỗi khung hình đến các khung hình được chọn
    # Kích thước (n, num_picks) hoặc (n,) nếu num_picks = 1
    # Đây là bước đầu tiên để tính min_{j∈S} ||x_i - x_j||^2 trong công thức (5)
    dist_mat = dist_mat[:,pick_idxs]
    
    # Tìm khoảng cách nhỏ nhất từ mỗi khung hình đến các khung hình được chọn
    # min_{j∈S} ||x_i - x_j||^2 với S là tập các khung hình được chọn
    # Điều này đo lường mức độ "gần" nhất mà mỗi khung hình trong video
    # có thể được đại diện bởi ít nhất một khung hình trong tóm tắt
    if num_picks > 1:
        # Tìm khoảng cách tối thiểu theo chiều thứ 1 (chiều của các khung hình được chọn)
        dist_mat = dist_mat.min(1, keepdim=True)[0]
    else:
        # Nếu chỉ có một khung hình được chọn, không cần tìm min
        dist_mat = dist_mat.view(-1, 1)
    
    # Tính phần thưởng đại diện (Công thức (5) trong paper)
    # R_rep = exp(-1/T·∑_{i=1}^T min_{j∈S}||x_i - x_j||^2)
    # Trong đó: 
    # - T là tổng số khung hình
    # - x_i là đặc trưng của khung hình i
    # - S = {i|a_i = 1, i = 1,2,...} là tập hợp các khung hình được chọn
    # - min_{j∈S}||x_i - x_j||^2 là khoảng cách nhỏ nhất từ khung i đến bất kỳ khung được chọn nào
    #
    # Trực quan: R_rep càng cao khi các khung hình được chọn càng gần với tất cả các khung hình khác
    # Mục tiêu: Khuyến khích chọn các khung hình đại diện tốt cho nội dung toàn bộ video
    #
    # Giải thích chi tiết:
    # 1. Tính trung bình khoảng cách tối thiểu: 1/T·∑_{i=1}^T min_{j∈S}||x_i - x_j||^2
    #    Đây là mức độ trung bình mà mỗi khung hình trong video được đại diện bởi tóm tắt
    # 2. Lấy exp(-) để chuyển khoảng cách thành độ tương đồng (càng xa càng nhỏ)
    #    Nếu khoảng cách trung bình nhỏ (tóm tắt đại diện tốt) thì R_rep sẽ cao
    reward_rep = torch.exp(-dist_mat.mean())
    
    # Đảm bảo reward_div và reward_rep ở cùng device với input
    reward_div = reward_div.to(device)
    reward_rep = reward_rep.to(device)
    
    # Kết hợp hai phần thưởng theo loại phần thưởng được yêu cầu
    # Các loại reward này được đánh giá trong Bảng 1 của paper, so sánh hiệu quả của từng loại
    if reward_type == 'dr':  # Sử dụng cả hai phần thưởng (DR-DSN)
        # R(S) = R_div + R_rep (Công thức (6) trong paper)
        # Trong paper, tác giả nhấn mạnh rằng: "R_div và R_rep bổ sung cho nhau và làm việc cùng nhau 
        # để hướng dẫn DSN" và "DR-DSN vượt trội hơn D-DSN và R-DSN trên cả hai tập dữ liệu"
        # 
        # Nhân với 0.5 để cân bằng giữa hai phần thưởng, giữ mức độ tổng thể tương tự
        reward = (reward_div + reward_rep) * 0.5
    elif reward_type == 'd':  # Chỉ sử dụng phần thưởng đa dạng (D-DSN)
        # Chỉ sử dụng R_div để đánh giá hiệu quả của riêng phần thưởng đa dạng
        # Trong Bảng 1, D-DSN cho kết quả tốt hơn R-DSN, nhưng không tốt bằng DR-DSN
        reward = reward_div
    elif reward_type == 'r':  # Chỉ sử dụng phần thưởng đại diện (R-DSN)
        # Chỉ sử dụng R_rep để đánh giá hiệu quả của riêng phần thưởng đại diện
        reward = reward_rep
    elif reward_type == 'd-nolambda':  # D-DSN không có lambda (không bỏ qua sự tương đồng xa về thời gian)
        # Đối với loại phần thưởng này, chúng ta cần tính lại reward_div với ignore_far_sim=False
        # Mục đích là để kiểm tra hiệu quả của quy tắc d(x_i, x_j) = 1 nếu |i - j| > λ
        # Đảm bảo chúng ta sử dụng tất cả các khung hình, không bỏ qua các khung xa về mặt thời gian
        
        # Nếu trước đó đã tính reward_div với ignore_far_sim=True, chúng ta cần tính lại
        if ignore_far_sim and num_picks > 1:
            # Tính lại ma trận dissim_submat mà không áp dụng kỹ thuật lambda
            normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
            dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())
            dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
            
            # KHÔNG sử dụng kỹ thuật lambda ở đây (ignore_far_sim = False)
            # Tính lại reward_div không có lambda
            reward_div_nolambda = dissim_submat.sum() / (num_picks * (num_picks - 1.))
            reward = reward_div_nolambda
        else:
            # Nếu đã tính với ignore_far_sim=False hoặc chỉ có 1 khung hình được chọn
            reward = reward_div
    else:
        raise ValueError("reward_type phải là 'dr', 'd', 'r', hoặc 'd-nolambda'")

    return reward
