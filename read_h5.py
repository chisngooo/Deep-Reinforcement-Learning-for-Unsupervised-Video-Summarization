import h5py

# Mở file H5
with h5py.File('datasets/eccv16_dataset_summe_google_pool5.h5', 'r') as f:
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