import h5py
import cv2
import os
import os.path as osp
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True, help="path to h5 result file")
parser.add_argument('-d', '--frm-dir', type=str, required=True, help="path to frame directory")
parser.add_argument('-i', '--idx', type=int, default=0, help="which key to choose")
parser.add_argument('--fps', type=int, default=30, help="frames per second")
parser.add_argument('--save-dir', type=str, default='log', help="directory to save")
parser.add_argument('--save-name', type=str, default='summary.mp4', help="video name to save (ends with .mp4)")
args = parser.parse_args()

def get_frame_dimensions(frm_dir):
    """Lấy kích thước của frame đầu tiên trong thư mục"""
    # Tìm frame đầu tiên
    frm_files = [f for f in os.listdir(frm_dir) if f.endswith('.jpg')]
    if not frm_files:
        print("Error: No frames found in the directory")
        return None, None
    
    # Sắp xếp để lấy frame đầu tiên
    frm_files.sort()
    first_frm_path = osp.join(frm_dir, frm_files[0])
    
    # Đọc frame để lấy kích thước
    frm = cv2.imread(first_frm_path)
    if frm is None:
        print(f"Error: Could not read the first frame: {first_frm_path}")
        return None, None
    
    height, width = frm.shape[:2]
    print(f"Original frame dimensions: {width}x{height}")
    return width, height

def frm2video(frm_dir, summary, vid_writer):
    """Tạo video từ các frame được chọn bởi summary"""
    frame_count = 0
    for idx, val in enumerate(summary):
        if val == 1:
            # Frame name dạng '000001.jpg'
            frm_name = str(idx+1).zfill(6) + '.jpg'
            frm_path = osp.join(frm_dir, frm_name)
            
            # Kiểm tra sự tồn tại của frame
            if not osp.exists(frm_path):
                print(f"Warning: Frame {frm_path} does not exist, skipping...")
                continue
            
            # Đọc frame
            frm = cv2.imread(frm_path)
            if frm is None:
                print(f"Warning: Failed to read frame {frm_path}, skipping...")
                continue
            
            # Không resize frame, giữ nguyên kích thước gốc
            vid_writer.write(frm)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
    
    return frame_count

if __name__ == '__main__':
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Lấy kích thước frame gốc
    width, height = get_frame_dimensions(args.frm_dir)
    if width is None or height is None:
        print("Error: Could not determine frame dimensions")
        exit(1)
    
    # Đường dẫn file đầu ra
    output_path = osp.join(args.save_dir, args.save_name)
    
    # Khởi tạo video writer với codec MP4V
    print(f"Creating video with dimensions: {width}x{height} and FPS: {args.fps}")
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    vid_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        args.fps,
        (width, height),
    )
    
    if not vid_writer.isOpened():
        print("Error: Failed to open video writer. Trying with XVID codec...")
        # Thử với XVID nếu MP4V không hoạt động
        avi_path = output_path.replace('.mp4', '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid_writer = cv2.VideoWriter(
            avi_path,
            fourcc,
            args.fps,
            (width, height),
        )
        
        if not vid_writer.isOpened():
            print("Error: Could not initialize video writer with any codec")
            exit(1)
            
        output_path = avi_path
    
    try:
        # Đọc file H5 chứa summary
        print(f"Reading summary from {args.path}")
        h5_res = h5py.File(args.path, 'r')
        keys = list(h5_res.keys())
        
        if not keys:
            print("Error: No keys found in the H5 file")
            vid_writer.release()
            exit(1)
            
        if args.idx >= len(keys):
            print(f"Error: Index {args.idx} out of range. Available indices: 0 to {len(keys)-1}")
            vid_writer.release()
            exit(1)
            
        # Lấy key dựa vào index
        key = keys[args.idx]
        print(f"Using key: {key}")
        
        # Lấy machine summary
        if 'machine_summary' not in h5_res[key]:
            print(f"Error: 'machine_summary' not found in key {key}")
            vid_writer.release()
            exit(1)
            
        summary = h5_res[key]['machine_summary'][...]
        h5_res.close()
        
        # Tạo video từ các frame
        print(f"Creating summary video for key: {key}")
        frame_count = frm2video(args.frm_dir, summary, vid_writer)
        
        if frame_count == 0:
            print("Warning: No frames were added to the video")
        else:
            print(f"Summary video created with {frame_count} frames")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Đóng video writer
        vid_writer.release()
        print(f"Video saved to: {output_path}")