import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ['DSN']

class DSN(nn.Module):
    """Deep Summarization Network - Mạng Tóm tắt Sâu
    
    Như mô tả trong paper phần "Proposed Approach" và "Deep Summarization Network", 
    DSN là một mạng học sâu được thiết kế để dự đoán xác suất quan trọng cho các khung hình video
    và lựa chọn các khung hình tạo thành tóm tắt video.
    """
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        """
        Khởi tạo mạng DSN (Deep Summarization Network) 
        
        Trong paper, DSN được thiết kế theo kiến trúc encoder-decoder:
        - Encoder: là CNN (trong paper dùng GoogleNet) đã được pretrained trên ImageNet
          để trích xuất đặc trưng từ các khung hình ({x_t}^T_t=1 với T là số lượng khung hình)
        - Decoder: là BiRNN (Bidirectional RNN) được thiết kế ở đây với LSTM
          (Long Short-Term Memory) để nắm bắt phụ thuộc dài
        
        Như Hình 1 trong paper cho thấy, DSN nhận một video Y_i và ra quyết định nhị phân
        về việc các phần nào của video được chọn làm tóm tắt S. DSN sẽ nhận phản hồi thông qua
        hàm phần thưởng R(S) dựa trên chất lượng tóm tắt (tính đa dạng và tính đại diện).
        
        Tham số:
            in_dim (int): Kích thước đầu vào của vector đặc trưng (mặc định: 1024 từ GoogLeNet)
            hid_dim (int): Kích thước lớp ẩn của RNN (mặc định: 256, như đề cập trong phần "Implementation details") 
            num_layers (int): Số lớp RNN (mặc định: 1)
            cell (str): Loại tế bào RNN ('lstm' hoặc 'gru') (mặc định: 'lstm')
                        Paper sử dụng LSTM để tăng cường khả năng nắm bắt phụ thuộc dài trong các khung hình
        """
        super().__init__()  # Gọi hàm khởi tạo của lớp cha
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        
        # Mô hình sử dụng mạng RNN hai chiều (bidirectional) để học các đặc trưng thời gian
        # Trong paper phần "Deep Summarization Network" có mô tả việc sử dụng BiRNN
        if cell == 'lstm':
            # Sử dụng LSTM (Long Short-Term Memory) - trong paper dùng LSTM để nắm bắt phụ thuộc dài
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            # Sử dụng GRU (Gated Recurrent Unit)
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
            
        # Lớp fully connected để chuyển đặc trưng RNN thành xác suất cho từng khung hình
        # hid_dim*2 vì sử dụng RNN hai chiều (BiRNN)
        # Như được mô tả trong paper, FC layer được kết nối với sigmoid để tạo ra xác suất
        self.fc = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        """
        Lan truyền xuôi qua mạng DSN
        
        Trong paper, phần "Deep Summarization Network", tiến trình hoạt động của DSN:
        1. Mạng nhận đặc trưng đã được trích xuất từ CNN pretrained ({x_t}^T_t=1)
        2. BiRNN xử lý chuỗi đặc trưng để nắm bắt đặc tính thời gian và ngữ cảnh
        3. Lớp fully connected kết hợp với sigmoid tạo ra xác suất p_t cho từng khung hình
        4. Từ xác suất này, chuỗi hành động nhị phân a_t được lấy mẫu theo phân phối Bernoulli
           (phần lấy mẫu này không được thực hiện ở đây mà trong main.py)
        
        Tham số:
            x: Tensor đầu vào, kích thước (batch_size, seq_len, in_dim)
               chứa các đặc trưng của khung hình video đã được trích xuất từ CNN
               
        Trả về:
            p: Tensor chứa xác suất quan trọng dự đoán cho mỗi khung hình, 
               kích thước (batch_size, seq_len, 1)
               Đây là p_t trong công thức (1) của paper
        """
        # Truyền qua mạng BiRNN
        # h có kích thước (batch_size, seq_len, hid_dim*2)
        # BiRNN được sử dụng để nắm bắt phụ thuộc thời gian trong cả hai chiều
        # Mỗi trạng thái ẩn h_t là sự nối của trạng thái ẩn xuôi và ngược
        h, _ = self.rnn(x)
        
        # Áp dụng lớp fully connected và hàm sigmoid để thu được 
        # xác suất lựa chọn cho các khung hình trong khoảng [0,1]
        # p_t = σ(W·h_t) [Công thức (1) trong paper]
        # Trong đó: 
        # - p_t là xác suất chọn khung hình t
        # - σ là hàm sigmoid
        # - W là tham số của lớp fully connected
        # - h_t là đặc trưng từ BiRNN tại thời điểm t
        #
        # Các xác suất này sau đó sẽ được sử dụng để lấy mẫu các hành động nhị phân
        # a_t ~ Bernoulli(p_t) [Công thức (2) trong paper]
        p = F.sigmoid(self.fc(h))
        return p