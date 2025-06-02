#!/bin/bash
# Script Bash để chạy thí nghiệm DSNsup và DR-DSNsup trên SumMe và TVSum datasets

# Định nghĩa các tham số
datasets=("summe" "tvsum")
splits=(0 1 2 3 4)

# Hàm để chạy một thí nghiệm cụ thể
run_experiment() {
    local dataset=$1
    local split=$2
    local model_type=$3    # "dsnsup" hoặc "dr-dsnsup"
    local model_name=$4    # "DSNsup" hoặc "DR-DSNsup"
    local sup_only=$5      # "--sup-only" hoặc ""
    
    # Tạo thư mục cho thí nghiệm này - tạo từng cấp một để tránh lỗi
    local save_dir="log/${model_type}-${dataset}-split${split}"
    mkdir -p "$save_dir"
    
    if [ ! -d "$save_dir" ]; then
        echo "ERROR: Không thể tạo thư mục ${save_dir}. Kiểm tra quyền truy cập."
        return 1
    fi
    
    echo "Running ${model_name} on ${dataset} split ${split}"
    python main.py -d "datasets/eccv16_dataset_${dataset}_google_pool5.h5" \
                  -s "datasets/${dataset}_splits.json" \
                  -m "${dataset}" \
                  --split-id "${split}" \
                  --supervised \
                  ${sup_only} \
                  --save-dir "${save_dir}" \
                  --verbose
                  
    # Kiểm tra kết quả trả về
    if [ $? -ne 0 ]; then
        echo "ERROR: Thí nghiệm ${model_name} trên ${dataset} split ${split} thất bại"
        return 1
    fi
    
    echo "Thí nghiệm ${model_name} trên ${dataset} split ${split} hoàn thành"
    return 0
}

# Hàm để thu thập và tính toán F1-scores
collect_scores() {
    local model_type=$1
    local dataset=$2
    
    # Thu thập các điểm F1 cho mỗi split
    local scores=()
    for split in "${splits[@]}"; do
        local log_file="log/${model_type}-${dataset}-split${split}/log_test.txt"
        if [ -f "$log_file" ]; then
            local f1=$(grep -oP "Average F-score \K[0-9.]+(?=%)" "$log_file")
            if [ -n "$f1" ]; then
                scores+=("$f1")
                echo "${model_type} trên ${dataset} (Split ${split}): F1-score = ${f1}%"
            else
                echo "${model_type} trên ${dataset} (Split ${split}): Không thể lấy F1-score" >&2
            fi
        else
            echo "${model_type} trên ${dataset} (Split ${split}): Không tìm thấy log file" >&2
        fi
    done
    
    # Tính trung bình F1-score nếu có kết quả
    if [ ${#scores[@]} -gt 0 ]; then
        local sum=0
        for score in "${scores[@]}"; do
            sum=$(echo "$sum + $score" | bc)
        done
        local avg=$(echo "scale=2; $sum / ${#scores[@]}" | bc)
        echo "${model_type} trên ${dataset}: Trung bình F1-score = ${avg}% (${#scores[@]}/5 splits)"
    else
        echo "${model_type} trên ${dataset}: Không có kết quả hợp lệ" >&2
    fi
}

# Tạo file tóm tắt kết quả
summary_file="supervised_experiment_summary.txt"
echo "Kết quả thí nghiệm supervised learning - $(date)" > "$summary_file"
echo "=============================================" >> "$summary_file"

# Chạy DSNsup trên SumMe và TVSum
for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
        run_experiment "$dataset" "$split" "DSNsup" "DSNsup" "--sup-only"
    done
done

# Chạy DR-DSNsup trên SumMe và TVSum
for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
        run_experiment "$dataset" "$split" "DR-DSNsup" "DR-DSNsup" ""
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
python calculate_avg_scores.py

echo -e "\nTóm tắt thí nghiệm đã được lưu vào $summary_file"
