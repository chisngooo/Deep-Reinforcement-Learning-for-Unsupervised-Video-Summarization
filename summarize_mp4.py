# summarize_mp4.py
"""
End-to-end video summarisation with DR-DSN family.
-------------------------------------------------
python summarize_mp4.py \
       --video  demo/movie.mp4 \
       --ckpt   checkpoints/dr-dsn-summe.pth.tar \
       --outdir output \
       --model-type dr          # dr | d | r | d-nolambda | drsup | dsnsup
"""

import os, os.path as osp, argparse, json, tempfile, shutil, cv2, tqdm, h5py
import numpy as np, torch, torchvision
from torchvision import transforms
from models import DSN
import vsum_tools                       # từ repo Kaiyang Zhou

# ---------------------- CLI ----------------------
ap = argparse.ArgumentParser()
ap.add_argument('--video', required=True)
ap.add_argument('--ckpt',  required=True)
ap.add_argument('--model-type', default='dr',
                choices=['dr','d','r','d-nolambda','drsup','dsnsup'])
ap.add_argument('--fps-out', type=int, default=30)
ap.add_argument('--outdir',  default='summary_out')
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)
base = osp.splitext(osp.basename(args.video))[0]

# ---------------------- 1. extract frames 2 fps ----------------------
tmp = tempfile.mkdtemp(prefix='vsum_')
frames_dir = osp.join(tmp,'frames'); os.makedirs(frames_dir, exist_ok=True)
print('[1] Extracting frames 2 fps …')
os.system(f'ffmpeg -loglevel error -i "{args.video}" -vf fps=2 "{frames_dir}/%06d.jpg"')

# ---------------------- 2. GoogLeNet-pool5 features ------------------
print('[2] Extracting GoogLeNet pool5 …')
device   = 'cuda' if torch.cuda.is_available() else 'cpu'
gnet     = torchvision.models.googlenet(pretrained=True)
gnet     = torch.nn.Sequential(*list(gnet.children())[:-1]).to(device).eval()
prep     = transforms.Compose([
           transforms.ToPILImage(), transforms.Resize(256),
           transforms.CenterCrop(224), transforms.ToTensor(),
           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

feat = []
with torch.no_grad():
    for jpg in tqdm.tqdm(sorted(os.listdir(frames_dir))):
        img = cv2.imread(osp.join(frames_dir,jpg))[:,:,::-1]
        x   = prep(img).unsqueeze(0).to(device)
        feat.append(gnet(x).squeeze().cpu().numpy())
feat = np.vstack(feat).astype('float32')            # (T,1024)
T    = len(feat)

# ---------------------- 3. fake HDF5 sample --------------------------
print('[3] Building temporary HDF5 …')
h5_feat = osp.join(tmp,'feat.h5')
seg     = 60                                         # 30 s ở 2 fps
cps     = np.array([[i,min(i+seg,T)] for i in range(0,T,seg)])
with h5py.File(h5_feat,'w') as f:
    grp = f.create_group('video')
    grp['features'] = feat
    grp['gtscore']  = np.zeros(T)
    grp['gtsummary']= np.zeros(T)
    grp['change_points'] = cps
    grp['n_frame_per_seg'] = np.diff(np.r_[cps[:,0],cps[-1,1]])
    grp['n_frames'] = np.array(T*15)                 # thời gian gốc ≈ *15
    grp['picks']    = np.arange(T)
    grp['user_summary'] = np.zeros((1,T))

split_json = osp.join(tmp,'split.json')
json.dump([{"train_keys":[],"test_keys":["video"]}], open(split_json,'w'))

# ---------------------- 4. load model & score ------------------------
print('[4] Loading DSN & predicting importance …')
checkpoint = torch.load(args.ckpt, map_location=device)
model = DSN(1024,256).to(device); model.eval()
model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})

with torch.no_grad():
    seq   = torch.tensor(feat).unsqueeze(0).to(device)
    probs = model(seq).cpu().squeeze().numpy()       # (T,)

# ---------------------- 5. knapsack summary -------------------------
print('[5] Generating machine summary …')
with h5py.File(h5_feat,'r') as f:
    cps = f['video/change_points'][...]
    nfps= f['video/n_frame_per_seg'][...].tolist()
    nfr = int(f['video/n_frames'][()])
    picks=f['video/picks'][...]

summary = vsum_tools.generate_summary(probs, cps, nfr, nfps, picks)

# ---------------------- 6. write result H5 (optional) ---------------
res_h5  = osp.join(tmp,'result.h5')
with h5py.File(res_h5,'w') as f:
    grp=f.create_group('video')
    grp['machine_summary']=summary
    grp['score']=probs

# ---------------------- 7. stitch video -----------------------------
print('[6] Writing final MP4 …')
w,h = cv2.imread(osp.join(frames_dir,'000001.jpg')).shape[1::-1]
four = cv2.VideoWriter_fourcc(*'mp4v')
out  = cv2.VideoWriter(osp.join(args.outdir,f'{base}_summary.mp4'),
                       four,args.fps_out,(w,h))
count=0
for idx,val in enumerate(summary):
    if val==1:
        fpath=osp.join(frames_dir,f'{idx+1:06d}.jpg')
        if osp.exists(fpath):
            out.write(cv2.imread(fpath)); count+=1
out.release()
print(f'✔ Done. Summary length {count} frames '
      f'({count/args.fps_out:.1f} s). Saved to {args.outdir}')

# ---------------------- 8. cleanup ----------------------------------
shutil.rmtree(tmp)
