#!/bin/bash
# Script Bash để chạy thí nghiệm DSNsup và DR-DSNsup trên SumMe và TVSum datasets

# Xác định lệnh Python thích hợp để sử dụng
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
elif command -v py &>/dev/null; then
    PYTHON_CMD="py"
else
    echo "ERROR: Không tìm thấy lệnh Python. Vui lòng cài đặt Python và đảm bảo nó nằm trong PATH."
    exit 1
fi

echo "Sử dụng lệnh Python: $PYTHON_CMD"

# Định nghĩa các tham số
datasets=("summe" "tvsum")
splits=(0 1 2 3 4)

# Hàm để chạy một thí nghiệm cụ thể
run_experiment() {
    local dataset=$1
    local split=$2
    local model_name=$3     # "DSNsup" hoặc "DR-DSNsup"
    local sup_only=$4       # "--sup-only" hoặc ""
    
    local dataset_file="datasets/eccv16_dataset_${dataset}_google_pool5.h5"
    local split_file="datasets/${dataset}_splits.json"
    local save_dir="log/${model_name}-${dataset}-split${split}"
    local model_path="${save_dir}/model_epoch60.pth.tar"
    
    # Tạo thư mục cho thí nghiệm này
    mkdir -p "$save_dir"
    
    if [ ! -d "$save_dir" ]; then
        echo "ERROR: Không thể tạo thư mục ${save_dir}. Kiểm tra quyền truy cập."
        return 1
    fi
    
    # Bước 1: Huấn luyện mô hình
    echo "Đang huấn luyện ${model_name} trên ${dataset} (Split ${split})"
    $PYTHON_CMD main.py -d "$dataset_file" \
                  -s "$split_file" \
                  -m "$dataset" \
                  --split-id "${split}" \
                  --supervised \
                  ${sup_only} \
                  --save-dir "$save_dir" \
                  --gpu 0 \
                  --verbose
                  
    # Kiểm tra kết quả huấn luyện
    if [ $? -ne 0 ]; then
        echo "ERROR: Huấn luyện ${model_name} trên ${dataset} split ${split} thất bại"
        return 1
    fi
    
    # Bước 2: Đánh giá mô hình (nếu checkpoint tồn tại)
    if [ -f "$model_path" ]; then
        echo "Đang đánh giá ${model_name} trên ${dataset} (Split ${split})"
        $PYTHON_CMD main.py -d "$dataset_file" \
                      -s "$split_file" \
                      -m "$dataset" \
                      --split-id "${split}" \
                      --supervised \
                      ${sup_only} \
                      --save-dir "$save_dir" \
                      --gpu 0 \
                      --evaluate \
                      --resume "$model_path" \
                      --verbose \
                      --save-results
        
        if [ $? -eq 0 ]; then
            echo "Thí nghiệm ${model_name} trên ${dataset} split ${split} hoàn thành"
        else
            echo "WARNING: Đánh giá ${model_name} trên ${dataset} split ${split} thất bại"
        fi
    else
        echo "WARNING: Không tìm thấy model checkpoint: $model_path"
        return 1
    fi
    
    return 0
}

# Hàm để thu thập và tính toán F1-scores
collect_scores() {
    local model_name=$1
    local dataset=$2
    
    # Thu thập các điểm F1 cho mỗi split
    local scores=()
    for split in "${splits[@]}"; do
        local log_file="log/${model_name}-${dataset}-split${split}/log_test.txt"
        if [ -f "$log_file" ]; then
            local f1=$(grep -oP "Average F-score \K[0-9.]+(?=%)" "$log_file")
            if [ -n "$f1" ]; then
                scores+=("$f1")
                echo "${model_name} trên ${dataset} (Split ${split}): F1-score = ${f1}%"
            else
                echo "${model_name} trên ${dataset} (Split ${split}): Không thể lấy F1-score" >&2
            fi
        else
            echo "${model_name} trên ${dataset} (Split ${split}): Không tìm thấy log file" >&2
        fi
    done
    
    # Tính trung bình F1-score nếu có kết quả
    if [ ${#scores[@]} -gt 0 ]; then
        local sum=0
        for score in "${scores[@]}"; do
            sum=$(echo "$sum + $score" | bc)
        done
        local avg=$(echo "scale=2; $sum / ${#scores[@]}" | bc)
        echo "${model_name} trên ${dataset}: Trung bình F1-score = ${avg}% (${#scores[@]}/5 splits)"
    else
        echo "${model_name} trên ${dataset}: Không có kết quả hợp lệ" >&2
    fi
}

# Tạo file tóm tắt kết quả
summary_file="supervised_experiment_summary.txt"
echo "Kết quả thí nghiệm supervised learning - $(date)" > "$summary_file"
echo "=============================================" >> "$summary_file"

# Chạy DSNsup trên SumMe và TVSum
echo "=== Bắt đầu huấn luyện DSNsup ==="
for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
        run_experiment "$dataset" "$split" "DSNsup" "--sup-only"
    done
done

# Chạy DR-DSNsup trên SumMe và TVSum  
echo "=== Bắt đầu huấn luyện DR-DSNsup ==="
for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
        run_experiment "$dataset" "$split" "DR-DSNsup" ""
    done
done

# Thu thập và hiển thị kết quả
echo ""
echo "=== Tóm tắt kết quả ==="
for dataset in "${datasets[@]}"; do
    collect_scores "DSNsup" "$dataset" | tee -a "$summary_file"
    collect_scores "DR-DSNsup" "$dataset" | tee -a "$summary_file"
done

echo ""
echo "Tính toán F1-scores chi tiết"
$PYTHON_CMD calculate_f1_scores.py

# Ghi thêm thông tin môi trường vào file tóm tắt để hỗ trợ debug
echo -e "\n=== Thông tin môi trường ===" >> "$summary_file"
echo "Lệnh Python sử dụng: $PYTHON_CMD" >> "$summary_file"
echo "Phiên bản Python: $($PYTHON_CMD --version 2>&1)" >> "$summary_file"
echo "Đường dẫn Python: $(which $PYTHON_CMD 2>&1)" >> "$summary_file"
echo "Hệ điều hành: $(uname -a 2>&1)" >> "$summary_file"
echo "Thời gian kết thúc: $(date)" >> "$summary_file"

echo -e "\nTóm tắt thí nghiệm đã được lưu vào $summary_file"
