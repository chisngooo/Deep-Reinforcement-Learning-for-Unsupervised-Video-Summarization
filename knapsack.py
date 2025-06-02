import numpy as np

'''
------------------------------------------------
Use dynamic programming (DP) to solve 0/1 knapsack problem
Time complexity: O(nW), where n is number of items and W is capacity

Author: Kaiyang Zhou
Website: https://kaiyangzhou.github.io/
------------------------------------------------
knapsack_dp(values,weights,n_items,capacity,return_all=False)

Input arguments:
  1. values: a list of numbers in either int or float, specifying the values of items
  2. weights: a list of int numbers specifying weights of items
  3. n_items: an int number indicating number of items
  4. capacity: an int number indicating the knapsack capacity
  5. return_all: whether return all info, defaulty is False (optional)

Return:
  1. picks: a list of numbers storing the positions of selected items
  2. max_val: maximum value (optional)
------------------------------------------------
'''
def knapsack_dp(values,weights,n_items,capacity,return_all=False):
    """
    Giải bài toán Knapsack 0/1 bằng quy hoạch động (dynamic programming)
    
    Như paper đề cập trong phần "Summary Generation":
    "The maximization step is essentially the 0/1 Knapsack problem, which is known as NP-hard. 
    We obtain a near-optimal solution via dynamic programming (Song et al. 2015)."
    
    Trong bài toán này:
    - values: điểm quan trọng của các đoạn (shot) video (trung bình của điểm quan trọng các khung hình)
    - weights: số khung hình trong mỗi đoạn (shot) video
    - n_items: số đoạn (shot) video
    - capacity: giới hạn số khung hình được chọn (thường là 15% tổng số khung hình)
    
    Mục tiêu: Chọn các đoạn (shot) video để tối đa hóa tổng điểm quan trọng
    nhưng không vượt quá giới hạn số khung hình cho phép.
    """
    check_inputs(values,weights,n_items,capacity)

    # Khởi tạo bảng DP
    # table[i,w]: giá trị tối đa khi xét i đoạn đầu tiên với giới hạn w khung hình
    table = np.zeros((n_items+1,capacity+1),dtype=np.float32)
    # keep[i,w]: 1 nếu đoạn thứ i được chọn với giới hạn w khung hình, 0 nếu không
    keep = np.zeros((n_items+1,capacity+1),dtype=np.float32)

    # Điền bảng DP theo công thức quy hoạch động chuẩn của bài toán Knapsack
    for i in range(1,n_items+1):
        for w in range(0,capacity+1):
            wi = weights[i-1] # weight of current item (số khung hình của đoạn i)
            vi = values[i-1] # value of current item (điểm quan trọng của đoạn i)
            
            # Nếu có thể chọn đoạn i (không vượt quá giới hạn w) và việc chọn
            # sẽ mang lại giá trị tốt hơn việc không chọn
            if (wi <= w) and (vi + table[i-1,w-wi] > table[i-1,w]):
                # Chọn đoạn i
                table[i,w] = vi + table[i-1,w-wi]
                keep[i,w] = 1
            else:
                # Không chọn đoạn i
                table[i,w] = table[i-1,w]

    # Truy vết để lấy các đoạn video được chọn
    picks = []  # Danh sách các đoạn được chọn
    K = capacity  # Công suất còn lại

    # Lặp từ đoạn cuối về đầu để tìm các đoạn được chọn
    for i in range(n_items,0,-1):
        if keep[i,K] == 1:  # Nếu đoạn i được chọn
            picks.append(i)  # Thêm vào danh sách
            K -= weights[i-1]  # Giảm công suất còn lại

    # Sắp xếp các đoạn theo thứ tự tăng dần
    picks.sort()
    # Chuyển từ chỉ số 1-index sang 0-index để phù hợp với Python
    picks = [x-1 for x in picks] # change to 0-index

    if return_all:
        max_val = table[n_items,capacity]
        return picks,max_val
    return picks

def check_inputs(values,weights,n_items,capacity):
    # check variable type
    assert(isinstance(values,list))
    assert(isinstance(weights,list))
    assert(isinstance(n_items,int))
    assert(isinstance(capacity,int))
    # check value type
    assert(all(isinstance(val,int) or isinstance(val,float) for val in values))
    assert(all(isinstance(val,int) for val in weights))
    # check validity of value
    assert(all(val >= 0 for val in weights))
    assert(n_items > 0)
    assert(capacity > 0)

if __name__ == '__main__':
    values = [2,3,4]
    weights = [1,2,3]
    n_items = 3
    capacity = 3
    picks = knapsack_dp(values,weights,n_items,capacity)
    print(picks)
