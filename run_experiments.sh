#!/bin/bash

# Script Bash để chạy thí nghiệm với các mô hình khác nhau (DR-DSN, R-DSN, D-DSN, D-DSN-nolambda)
# trên hai bộ dữ liệu (SumMe và TVSum) và tính toán các chỉ số F1 trung bình

# Định nghĩa các tham số
datasets=("summe" "tvsum")
splits=(0 1 2 3 4)  # 5 splits khác nhau

# Các loại reward và tên mô hình tương ứng
declare -A reward_types
reward_types["dr"]="DR-DSN"
reward_types["d"]="D-DSN"
reward_types["r"]="R-DSN"
reward_types["d-nolambda"]="D-DSN-nolambda"

# Hàm để chạy một thí nghiệm cụ thể
run_experiment() {
    local dataset=$1
    local split=$2
    local reward_type=$3
    local model_name=$4
    local train=$5  # true hoặc false
    local evaluate=$6  # true hoặc false
    
    local dataset_file="datasets/eccv16_dataset_${dataset}_google_pool5.h5"
    local split_file="datasets/${dataset}_splits.json"
    local save_dir="log/${model_name}-${dataset}-split${split}"
    local model_path="${save_dir}/model_epoch60.pth.tar"
    
    # Tạo thư mục nếu chưa tồn tại
    mkdir -p "$save_dir"

    # Huấn luyện mô hình nếu yêu cầu
    if [ "$train" = true ]; then
        echo "Đang huấn luyện ${model_name} trên ${dataset} (Split ${split})"
        python main.py -d "$dataset_file" -s "$split_file" -m "$dataset" --save-dir "$save_dir" --gpu 0 --split-id "$split" --verbose --reward-type "$reward_type"
    fi

    # Đánh giá mô hình nếu yêu cầu và mô hình đã tồn tại
    if [ "$evaluate" = true ] && [ -f "$model_path" ]; then
        echo "Đang đánh giá ${model_name} trên ${dataset} (Split ${split})"
        python main.py -d "$dataset_file" -s "$split_file" -m "$dataset" --save-dir "$save_dir" --gpu 0 --split-id "$split" --evaluate --resume "$model_path" --verbose --save-results --reward-type "$reward_type"
    elif [ "$evaluate" = true ]; then
        echo "Bỏ qua đánh giá cho ${model_name} trên ${dataset} (Split ${split}): Không tìm thấy mô hình"
    fi
}

# Hàm để trích xuất F1-score từ log file
get_f1_score() {
    local log_file=$1
    
    if [ -f "$log_file" ]; then
        f1=$(grep -oP "Average F-score \K[0-9.]+(?=%)" "$log_file")
        if [ -n "$f1" ]; then
            echo "scale=4; $f1/100" | bc
            return 0
        fi
    fi
    return 1
}

# Tạo một file để lưu trữ kết quả
summary_file="experiment_summary.txt"
echo "Kết quả Thí nghiệm - $(date)" > "$summary_file"

# Chạy thí nghiệm cho mỗi tổ hợp reward type, dataset và split
for reward_type in "${!reward_types[@]}"; do
    model_name=${reward_types["$reward_type"]}
    
    for dataset in "${datasets[@]}"; do
        results_key="${model_name}-${dataset}"
        declare -a scores
        
        for split in "${splits[@]}"; do
            # Chạy thí nghiệm
            run_experiment "$dataset" "$split" "$reward_type" "$model_name" true true
            
            # Thu thập kết quả
            log_file="log/${model_name}-${dataset}-split${split}/log_test.txt"
            f1_score=$(get_f1_score "$log_file")
            
            if [ -n "$f1_score" ]; then
                scores+=("$f1_score")
                echo "${model_name} trên ${dataset} (Split ${split}): F1-score = $f1_score"
            else
                echo "${model_name} trên ${dataset} (Split ${split}): Không thể lấy F1-score" >&2
            fi
        done
        
        # Tính trung bình F1-score nếu có kết quả
        if [ ${#scores[@]} -gt 0 ]; then
            sum=0
            for score in "${scores[@]}"; do
                sum=$(echo "$sum + $score" | bc)
            done
            avg=$(echo "scale=4; $sum / ${#scores[@]}" | bc)
            formatted=$(printf "%.2f%%" $(echo "$avg * 100" | bc))
            echo "${results_key}: $formatted (Splits: ${#scores[@]}/5)"
            
            # Ghi kết quả vào file
            echo "${results_key}: $formatted (Điểm số riêng lẻ: ${scores[*]})" >> "$summary_file"
        else
            echo "${results_key}: Không có kết quả hợp lệ" >&2
            echo "${results_key}: Không có kết quả hợp lệ" >> "$summary_file"
        fi
    done
done

echo -e "\nTóm tắt thí nghiệm đã được lưu vào $summary_file"
