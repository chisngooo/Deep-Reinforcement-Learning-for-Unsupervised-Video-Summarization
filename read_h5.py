import h5py
import os

# Use absolute path to ensure correct file location
file_path = 'datasets/TowerScratch-v0_975.h5'
abs_path = os.path.abspath(file_path)

# Check if file exists before trying to open it
if os.path.exists(abs_path):
    print(f"File found at: {abs_path}")
    try:
        with h5py.File(abs_path, 'r') as f:
            # Liệt kê các nhóm cấp cao nhất
            print("Các nhóm trong file:")
            for key in f.keys():
                print(f"- {key}")
                
            # In thông tin chi tiết hơn về cấu trúc
            def print_attrs(name, obj):
                print(f"{name} có kiểu: {type(obj)}")
                if isinstance(obj, h5py.Dataset):
                    print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")
            
            # Duyệt qua từng phần tử trong file
            f.visititems(print_attrs)
    except Exception as e:
        print(f"Error opening file: {e}")
else:
    print(f"File not found: {abs_path}")
    print("Current working directory:", os.getcwd())
    print("Files in datasets directory:")
    datasets_dir = os.path.join(os.getcwd(), 'datasets')
    if os.path.exists(datasets_dir):
        print(os.listdir(datasets_dir))
    else:
        print(f"'datasets' directory not found in {os.getcwd()}")