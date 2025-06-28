#!/bin/bash
# ==============================================================================
# Script Bash để đánh giá tất cả 60 model chưa qua training trên datasets SumMe và TVSum
# Mục đích: Đánh giá baseline performance với random initialization 
# Tương tự như run_experiments.sh và run_supervised.sh nhưng không train
# ==============================================================================

# Phân tích các tham số dòng lệnh
VERBOSE_MODE=false
USE_GPU=true
CUSTOM_SEED=""
CUSTOM_OUTPUT=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --verbose|-v) VERBOSE_MODE=true ;;
        --no-gpu) USE_GPU=false ;;
        --seed) CUSTOM_SEED="$2"; shift ;;
        --output|-o) CUSTOM_OUTPUT="$2"; shift ;;
        --help|-h) 
            echo "Cách sử dụng: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --verbose, -v       Hiển thị chi tiết kết quả của từng video"
            echo "  --no-gpu            Chạy trên CPU thay vì GPU"
            echo "  --seed VALUE        Đặt random seed (mặc định: 7)"
            echo "  --output, -o FILE   Đặt tên file output"
            echo "  --help, -h          Hiển thị hướng dẫn này"
            exit 0
            ;;
        *) echo "Tham số không hợp lệ: $1"; exit 1 ;;
    esac
    shift
done

# Hiển thị banner
echo "======================================================================"
echo "  ĐÁNH GIÁ 60 MÔ HÌNH CHƯA QUA TRAINING (RANDOM INITIALIZATION)"
echo "  - 6 kiến trúc × 2 datasets × 5 splits"
echo "======================================================================"
echo ""

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

# Kiểm tra các dependencies cần thiết
echo "Kiểm tra dependencies..."
REQUIRED_PACKAGES=("torch" "numpy" "h5py" "tabulate")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! $PYTHON_CMD -c "import $package" &>/dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "CẢNH BÁO: Thiếu các package Python sau: ${MISSING_PACKAGES[*]}"
    echo "Vui lòng cài đặt chúng bằng lệnh:"
    echo "pip install ${MISSING_PACKAGES[*]}"
    
    read -p "Bạn có muốn tiếp tục không? (y/n): " continue_choice
    if [[ $continue_choice != "y" && $continue_choice != "Y" ]]; then
        exit 1
    fi
fi

# Kiểm tra GPU nếu yêu cầu
if [ "$USE_GPU" = true ]; then
    if ! $PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        echo "CẢNH BÁO: Không tìm thấy GPU hoặc CUDA không khả dụng!"
        echo "Khuyến nghị sử dụng CPU với tham số --no-gpu."
        read -p "Bạn vẫn muốn tiếp tục (có thể gây lỗi)? (y/n): " continue_gpu
        if [[ $continue_gpu != "y" && $continue_gpu != "Y" ]]; then
            USE_GPU=false
            echo "Chuyển sang chạy trên CPU."
        fi
    else
        GPU_INFO=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        MEMORY_INFO=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / (1024**3))" 2>/dev/null)
        echo "GPU phát hiện: $GPU_INFO (${MEMORY_INFO:.1f} GB)"
    fi
fi

# Định nghĩa các tham số
datasets=("summe" "tvsum")
splits=(0 1 2 3 4)
model_types=("DR-DSN" "D-DSN" "R-DSN" "D-DSN-nolambda" "DR-DSNsup" "DSNsup")

# Sử dụng fixed seed để đảm bảo reproducibility
SEED=${CUSTOM_SEED:-7}

# Thiết lập file đầu ra
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "$CUSTOM_OUTPUT" ]; then
    OUTPUT_FILE="$CUSTOM_OUTPUT"
else
    OUTPUT_FILE="untrained_model_evaluation_$TIMESTAMP.json"
fi
echo "Kết quả sẽ được lưu vào: $OUTPUT_FILE"

# Kiểm tra datasets có tồn tại không
for dataset in "${datasets[@]}"; do
    h5_path="datasets/eccv16_dataset_${dataset}_google_pool5.h5"
    split_path="datasets/${dataset}_splits.json"
    
    if [ ! -f "$h5_path" ]; then
        echo "ERROR: Dataset file không tồn tại: $h5_path"
        echo "Vui lòng kiểm tra đường dẫn datasets/"
        exit 1
    fi
    
    if [ ! -f "$split_path" ]; then
        echo "ERROR: Split file không tồn tại: $split_path"
        echo "Vui lòng kiểm tra đường dẫn datasets/"
        exit 1
    fi
done

# Hiển thị tổng quan về công việc
echo ""
echo "=== TỔNG QUAN CÔNG VIỆC ==="
echo "- 6 kiến trúc mô hình: ${model_types[*]}"
echo "- 2 datasets: ${datasets[*]}"
echo "- 5 splits mỗi dataset: ${splits[*]}"
echo "- Random seed: $SEED"
echo "- Tổng cộng 60 phép đánh giá"
echo "- Sử dụng GPU: $USE_GPU"
echo "- Verbose mode: $VERBOSE_MODE"

# Ước tính thời gian chạy (khoảng 2-3 phút cho mỗi model với GPU, lâu hơn với CPU)
TOTAL_MODELS=60
if [ "$USE_GPU" = true ]; then
    EST_TIME=$((TOTAL_MODELS * 2))  # ~2 phút mỗi model với GPU
else
    EST_TIME=$((TOTAL_MODELS * 5))  # ~5 phút mỗi model với CPU
fi
echo "- Ước tính thời gian chạy: ~$EST_TIME phút"
echo ""

# Xác nhận với người dùng trước khi bắt đầu
read -p "Bắt đầu đánh giá? (y/n): " start_confirm
if [[ $start_confirm != "y" && $start_confirm != "Y" ]]; then
    echo "Hủy đánh giá."
    exit 0
fi

# Tạo thư mục kết quả nếu chưa có
mkdir -p "results"

# Thiết lập các tham số cho Python script
PYTHON_ARGS=("--output" "$OUTPUT_FILE" "--seed" "$SEED")
if [ "$VERBOSE_MODE" = true ]; then
    PYTHON_ARGS+=("--verbose")
fi
if [ "$USE_GPU" = false ]; then
    PYTHON_ARGS+=("--no-gpu")
fi

# Chạy đánh giá cho tất cả 60 model (6 kiến trúc x 2 datasets x 5 splits)
echo "Bắt đầu đánh giá 60 model untrained..."
$PYTHON_CMD evaluate_untrained_models.py "${PYTHON_ARGS[@]}"

# Đánh dấu thời gian bắt đầu để tính thời gian chạy
START_TIME=$(date +%s)

# Kiểm tra kết quả
if [ $? -eq 0 ]; then
    echo "✅ Đánh giá thành công. Kết quả đã được lưu vào: $OUTPUT_FILE"
else
    echo "❌ ERROR: Đánh giá thất bại!"
    echo "   Vui lòng kiểm tra lỗi ở trên và thử lại."
    echo "   Nếu lỗi liên quan đến GPU memory, hãy thử chạy trên CPU với: --no-gpu"
    exit 1
fi

# Tính thời gian chạy
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
MINUTES=$((RUNTIME / 60))
SECONDS=$((RUNTIME % 60))

echo ""
echo "⏱️  Thời gian chạy: $MINUTES phút $SECONDS giây"
echo ""

# Tạo thư mục results nếu chưa tồn tại
mkdir -p "results"

# Tạo bảng so sánh trained vs untrained từ file JSON
echo "Tạo báo cáo so sánh từ kết quả..."
REPORT_FILE="results/untrained_models_summary_$TIMESTAMP.txt"

# Chạy phân tích và tạo báo cáo chi tiết
$PYTHON_CMD -c "
import json
import numpy as np
from tabulate import tabulate
import os
from datetime import datetime

# Thiết lập file báo cáo
report_file = '$REPORT_FILE'

# Tạo header cho file báo cáo
with open(report_file, 'w') as f:
    f.write('=' * 80 + '\n')
    f.write('BÁO CÁO ĐÁNH GIÁ MÔ HÌNH CHƯA QUA TRAINING\n')
    f.write('=' * 80 + '\n')
    f.write(f'Thời gian: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\n')
    f.write(f'File kết quả: {os.path.basename(\"$OUTPUT_FILE\")}\n')
    f.write(f'Random seed: $SEED\n\n')

# Đọc kết quả từ file
try:
    with open('$OUTPUT_FILE', 'r') as f:
        results = json.load(f)
    
    # Tạo bảng so sánh performance
    table = []
    for model_type in ['DR-DSN', 'D-DSN', 'R-DSN', 'D-DSN-nolambda', 'DR-DSNsup', 'DSNsup']:
        for dataset in ['summe', 'tvsum']:
            if model_type in results and dataset in results[model_type]:
                avg_score = results[model_type][dataset]['average']['mean']
                std_score = results[model_type][dataset]['average']['std']
                split_scores = results[model_type][dataset]['average']['splits']
                split_str = ', '.join([f'S{i}: {s:.1%}' for i, s in enumerate(split_scores)])
                
                table.append([
                    model_type,
                    dataset.upper(),
                    f'{avg_score:.1%} ± {std_score:.1%}',
                    split_str
                ])
    
    # In và lưu bảng
    summary_table = tabulate(table, headers=['Model', 'Dataset', 'F1-Score (mean ± std)', 'Split Details'], tablefmt='grid')
    print('\nKẾT QUẢ ĐÁNH GIÁ UNTRAINED MODELS:')
    print(summary_table)
    
    with open(report_file, 'a') as f:
        f.write('BẢNG KẾT QUẢ ĐÁNH GIÁ\n')
        f.write('-' * 80 + '\n')
        f.write(summary_table)
        f.write('\n\n')
    
    # Thêm thông tin so sánh với các model đã được train (từ paper)
    trained_scores = {
        'DR-DSN': {'summe': 0.399, 'tvsum': 0.566},
        'D-DSN': {'summe': 0.395, 'tvsum': 0.557},
        'R-DSN': {'summe': 0.388, 'tvsum': 0.566},
        'DR-DSNsup': {'summe': 0.419, 'tvsum': 0.567},
        'DSNsup': {'summe': 0.392, 'tvsum': 0.523},
        'D-DSN-nolambda': {'summe': 0.392, 'tvsum': 0.523}
    }
    
    # Tạo bảng so sánh
    comparison = []
    for model_type in trained_scores:
        for dataset in ['summe', 'tvsum']:
            if model_type in results and dataset in results[model_type]:
                untrained = results[model_type][dataset]['average']['mean']
                trained = trained_scores[model_type][dataset]
                improvement = trained - untrained
                rel_improvement = (improvement / untrained) * 100 if untrained > 0 else 0
                
                comparison.append([
                    model_type,
                    dataset.upper(),
                    f'{untrained:.1%}',
                    f'{trained:.1%}',
                    f'+{improvement:.1%}',
                    f'+{rel_improvement:.1f}%'
                ])
    
    # In và lưu bảng so sánh
    comparison_table = tabulate(comparison, 
                              headers=['Model', 'Dataset', 'Untrained', 'Trained', 'Abs. Improvement', 'Rel. Improvement'],
                              tablefmt='grid')
    print('\nSO SÁNH VỚI TRAINED MODELS:')
    print(comparison_table)
    
    with open(report_file, 'a') as f:
        f.write('SO SÁNH VỚI TRAINED MODELS\n')
        f.write('-' * 80 + '\n')
        f.write(comparison_table)
        f.write('\n\n')
    
    # Thêm các thống kê và phân tích
    with open(report_file, 'a') as f:
        # Tính toán trung bình cải thiện
        improvements = []
        for model_type in trained_scores:
            for dataset in ['summe', 'tvsum']:
                if model_type in results and dataset in results[model_type]:
                    untrained = results[model_type][dataset]['average']['mean']
                    trained = trained_scores[model_type][dataset]
                    improvements.append(trained - untrained)
        
        avg_improvement = np.mean(improvements)
        f.write('THỐNG KÊ VÀ PHÂN TÍCH\n')
        f.write('-' * 80 + '\n')
        f.write(f'Cải thiện trung bình sau training: +{avg_improvement:.1%}\n')
        f.write(f'Số lượng model được đánh giá: {len(table)}/12\n')
        f.write(f'Số lượng model được so sánh: {len(comparison)}/12\n')
        f.write(f'Thời gian chạy: {$MINUTES} phút {$SECONDS} giây\n\n')
    
    print(f'\nBáo cáo chi tiết đã được lưu vào: {report_file}')
    
except Exception as e:
    print(f'Lỗi khi xử lý kết quả: {str(e)}')
    with open(report_file, 'a') as f:
        f.write(f'ERROR: {str(e)}\n')
" 2> /dev/null || echo "⚠️ Lỗi khi tạo báo cáo - kiểm tra file $OUTPUT_FILE để xem kết quả thô"

# Tạo bản sao của file output với tên dễ nhớ hơn
cp "$OUTPUT_FILE" "results/untrained_model_results_latest.json"
echo "Đã tạo bản sao của kết quả tại: results/untrained_model_results_latest.json"

echo ""
echo "✨ Hoàn thành! ✨"
echo "- Kết quả đầy đủ: $OUTPUT_FILE"
echo "- Báo cáo tóm tắt: $REPORT_FILE"
echo "- Bản sao kết quả: results/untrained_model_results_latest.json"
