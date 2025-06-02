import cv2, os
vid = cv2.VideoCapture('pytorch-vsumm-reinforce/video/dv2.mp4')

# Lấy thông tin FPS của video
fps = vid.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

# Lấy thêm thông tin khác của video
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
duration_seconds = total_frames / fps if fps > 0 else 0

print(f"Resolution: {width}x{height}")
print(f"Total frames: {total_frames}")
print(f"Duration: {duration_seconds:.2f} seconds ({duration_seconds/60:.2f} minutes)")

out_dir = 'pytorch-vsumm-reinforce/video/dv2_frames'
os.makedirs(out_dir, exist_ok=True)
i = 1
while True:
    ret, frame = vid.read()
    if not ret: break
    fname = os.path.join(out_dir, f'{i:06d}.jpg')
    cv2.imwrite(fname, frame)
    i += 1

print(f"Extracted {i-1} frames")