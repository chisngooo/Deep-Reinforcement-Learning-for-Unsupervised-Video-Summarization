"""
AI Video Summarization - Beautiful UI with Hierarchical Model Selection
"""

import streamlit as st
import os, os.path as osp, json, tempfile, shutil, cv2, h5py
import numpy as np, torch, torchvision
from torchvision import transforms
from models import DSN
import vsum_tools
import subprocess
from PIL import Image
import glob
# Add visualization imports
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

from vsum_tools import calc_kernel, smart_kts, cpd_auto       # KTS functions
def get_available_models():
    """Get all available models organized by type, dataset, and split"""
    log_dir = "log"
    models = {}
    
    if not osp.exists(log_dir):
        return models
    
    # Get all model directories
    model_dirs = [d for d in os.listdir(log_dir) if osp.isdir(osp.join(log_dir, d))]
    
    for model_dir in model_dirs:
        parts = model_dir.split('-')
        if len(parts) >= 3:
            # Parse model directory name
            model_type = '-'.join(parts[:-2])  # e.g., "DR-DSN", "DR-DSNsup"
            dataset = parts[-2]  # e.g., "tvsum", "summe"
            split = parts[-1]    # e.g., "split0", "split1"
            
            # Check if model file exists
            model_path = osp.join(log_dir, model_dir, "model_epoch60.pth.tar")
            if osp.exists(model_path):
                if model_type not in models:
                    models[model_type] = {}
                if dataset not in models[model_type]:
                    models[model_type][dataset] = []
                models[model_type][dataset].append({
                    'split': split,
                    'path': model_path,
                    'dir': model_dir
                })
    
    # Sort splits for each model type and dataset
    for model_type in models:
        for dataset in models[model_type]:
            models[model_type][dataset].sort(key=lambda x: x['split'])
    
    return models

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def create_web_compatible_video(input_path, output_path, fps=30):
    """Create web-compatible MP4 video using ffmpeg with H.264"""
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', input_path,
        '-c:v', 'libx264',          # H.264 codec (most compatible)
        '-profile:v', 'baseline',    # Baseline profile for max compatibility 
        '-level', '3.0',            # Level 3.0 for web compatibility
        '-pix_fmt', 'yuv420p',      # Pixel format compatible with all browsers
        '-movflags', '+faststart',   # Enable fast start for web streaming
        '-preset', 'medium',        # Encoding preset
        '-crf', '23',               # Quality setting (18-28, lower = better quality)
        '-r', str(fps),             # Frame rate
        output_path
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return True, "Success"
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "FFmpeg timeout"
    except Exception as e:
        return False, str(e)

def summarize_video(video_path, model_path, fps_out=30, progress_callback=None, status_callback=None):
    """
    Video summarization function based on summarize_mp4.py
    Returns: (success, output_path, stats, error_message)
    """
    try:
        # Create temporary directory
        tmp = tempfile.mkdtemp(prefix='streamlit_vsum_')
        frames_dir = osp.join(tmp, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        base = osp.splitext(osp.basename(video_path))[0]
        
        # ---------------------- 1. Extract frames 2 fps ----------------------
        if status_callback:
            status_callback("Step 1/6: Extracting frames at 2 FPS...")
        if progress_callback:
            progress_callback(10)
            
        cmd = f'ffmpeg -loglevel error -i "{video_path}" -vf fps=2 "{frames_dir}/%06d.jpg"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return False, None, None, f"FFmpeg frame extraction failed: {result.stderr}"
        
        # Check if frames were extracted
        frame_files = sorted(os.listdir(frames_dir))
        if not frame_files:
            return False, None, None, "No frames were extracted from the video"
        
        # ---------------------- 2. GoogLeNet-pool5 features ------------------
        if status_callback:
            status_callback("Step 2/6: Extracting GoogLeNet pool5 features...")
        if progress_callback:
            progress_callback(25)
            
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gnet = torchvision.models.googlenet(pretrained=True)
        gnet = torch.nn.Sequential(*list(gnet.children())[:-1]).to(device).eval()
        prep = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize(256),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        
        feat = []
        with torch.no_grad():
            for i, jpg in enumerate(frame_files):
                img = cv2.imread(osp.join(frames_dir, jpg))[:,:,::-1]  # BGR to RGB
                x = prep(img).unsqueeze(0).to(device)
                feat.append(gnet(x).squeeze().cpu().numpy())
                  # Update progress
                if i % 10 == 0 and progress_callback:
                    progress = 25 + (i / len(frame_files)) * 20
                    progress_callback(int(progress))
        
        feat = np.vstack(feat).astype('float32')  # (T,1024)
        T = len(feat)
        
        # Create temporary HDF5 file path
        h5_feat = osp.join(tmp, f'{base}_features.h5')        # ---------------------- 3. Shot segmentation via KTS -------------------
        if status_callback:
            status_callback("Step 3/6: Segmenting shots with KTS ...")
        if progress_callback:
            progress_callback(45)

        # 3a. Improved KTS with multiple fallback options
        granularity = 20                  # 20 khung 2 fps  →  10 s gốc
        max_ncp    = min(100, T // granularity)  # paper đặt trần 100
        
        # Use smart KTS that automatically selects best method
        change_points = smart_kts(feat, max_ncp=max_ncp, method="auto")

        # 3c. FIXED: chuyển thành [start,end] với indexing nhất quán
        # Đảm bảo change_points được sắp xếp và trong phạm vi hợp lệ
        if len(change_points) > 0:
            change_points = np.sort(change_points)
            change_points = change_points[(change_points > 0) & (change_points < T)]

        # Tạo segments với end index là EXCLUSIVE (chuẩn Python)
        segment_boundaries = np.concatenate(([0], change_points, [T]))
        cps = []
        for i in range(len(segment_boundaries) - 1):
            start = int(segment_boundaries[i])
            end = int(segment_boundaries[i + 1])
            if end > start:  # Đảm bảo segment hợp lệ
                cps.append([start, end - 1])  # Convert to inclusive end for storage

        cps = np.array(cps, dtype=int)

        # Debug information
        print(f"DEBUG KTS: T={T}, change_points={change_points}")
        print(f"DEBUG KTS: segments shape={cps.shape}")
        print(f"DEBUG KTS: segments=\n{cps}")

        # 3d. FIXED: tính n_frame_per_seg với indexing nhất quán  
        n_frame_per_seg = []
        for i in range(len(cps)):
            start, end = cps[i]
            # Since cps stores inclusive end, length is (end - start + 1)
            length = end - start + 1
            n_frame_per_seg.append(length)

        n_frame_per_seg = np.array(n_frame_per_seg, dtype=int)

        # Verify segments don't exceed bounds
        print(f"DEBUG KTS: n_frame_per_seg={n_frame_per_seg}")
        print(f"DEBUG KTS: total frames in segments={np.sum(n_frame_per_seg)}, expected={T}")

        # 3e. ghi HDF5 tạm
        with h5py.File(h5_feat, 'w') as f:
            grp = f.create_group('video')
            grp['features']          = feat
            grp['gtscore']           = np.zeros(T)
            grp['gtsummary']         = np.zeros(T)
            grp['change_points']     = cps                    # (m,2) inclusive end
            grp['n_frame_per_seg']   = n_frame_per_seg       # Correct lengths
            grp['n_frames']          = np.array(T)           # đơn vị 2 fps
            grp['picks']             = np.arange(T)
            grp['user_summary']      = np.zeros((1, T))


        
        # ---------------------- 4. Load model & score ------------------------
        if status_callback:
            status_callback("Step 4/6: Loading DSN & predicting importance...")
        if progress_callback:
            progress_callback(65)
            
        if not osp.exists(model_path):
            return False, None, None, f"Model checkpoint not found: {model_path}"
        
        checkpoint = torch.load(model_path, map_location=device)
        model = DSN(1024, 256).to(device)
        model.eval()
        model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
        with torch.no_grad():
            seq = torch.tensor(feat).unsqueeze(0).to(device)
            probs = model(seq).cpu().squeeze().numpy()  # (T,)
          # ---------------------- 5. Knapsack summary -------------------------
        if status_callback:
            status_callback("Step 5/6: Generating machine summary...")
        if progress_callback:
            progress_callback(80)
            
        with h5py.File(h5_feat, 'r') as f:
            cps = f['video/change_points'][...]
            nfps = f['video/n_frame_per_seg'][...].tolist()
            nfr = int(f['video/n_frames'][()])
            picks = f['video/picks'][...]        
            summary = vsum_tools.generate_summary(probs, cps, nfr, nfps, picks, 
                                             proportion=summary_proportion/100.0,
                                             method=summary_method)
        
        # Debug the summary generation
        print(f"DEBUG Knapsack: probs shape: {probs.shape}")
        print(f"DEBUG Knapsack: probs min/max: {probs.min():.4f}/{probs.max():.4f}")
        print(f"DEBUG Knapsack: summary shape: {summary.shape}")
        print(f"DEBUG Knapsack: summary unique values: {np.unique(summary, return_counts=True)}")
        print(f"DEBUG Knapsack: Total selected frames: {np.sum(summary)}")
        
        # Remove temporal analysis - not part of original paper
        temporal_analysis = None
        
        # If all frames are selected, there might be an issue with the model or knapsack
        if np.sum(summary) == len(summary):
            print("WARNING: All frames were selected! Model may need adjustment.")
        elif np.sum(summary) == 0:
            print("WARNING: No frames were selected! Setting default selection.")
            # Select every 10th frame as fallback
            summary = np.zeros_like(summary)
            summary[::10] = 1
          # Create visualization data
        viz_fig = create_video_analysis_visualization(probs, summary, temporal_analysis)
        
        # ---------------------- 6. Create temp video first ------------------
        if status_callback:
            status_callback("Step 6/6: Creating summary video...")
        if progress_callback:
            progress_callback(90)
            
        # Get video dimensions
        sample_img = cv2.imread(osp.join(frames_dir, '000001.jpg'))
        h, w = sample_img.shape[:2]
        
        # Create temporary video with OpenCV (mp4v codec)
        temp_video_path = osp.join(tmp, f'{base}_temp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps_out, (w, h))
        
        count = 0
        for idx, val in enumerate(summary):
            if val == 1:
                fpath = osp.join(frames_dir, f'{idx+1:06d}.jpg')
                if osp.exists(fpath):
                    frame = cv2.imread(fpath)
                    out.write(frame)
                    count += 1
        out.release()
        
        if count == 0:
            return False, None, None, "No frames selected for summary"
        
        # ---------------------- 7. Convert to web-compatible format -----------
        output_dir = "streamlit_output"
        os.makedirs(output_dir, exist_ok=True)
        final_output_path = osp.join(output_dir, f'{base}_summary.mp4')
        
        # Convert to web-compatible format using FFmpeg
        success, error_msg = create_web_compatible_video(temp_video_path, final_output_path, fps_out)
        
        if not success:
            # If FFmpeg conversion fails, try to copy the temp file
            try:
                shutil.copy2(temp_video_path, final_output_path)
                success = True
                error_msg = "Using fallback video format"
            except Exception as e:
                return False, None, None, f"Video conversion failed: {error_msg}, Copy failed: {str(e)}"        # Calculate statistics with better accuracy
        original_duration = T / 2.0  # Since we extracted at 2 FPS
        summary_duration = count / fps_out
        
        # Debug output to check values
        print(f"DEBUG: T={T}, count={count}, fps_out={fps_out}")
        print(f"DEBUG: original_duration={original_duration}, summary_duration={summary_duration}")
        print(f"DEBUG: Summary ratio: {count/T:.3f}")
        
        # Fixed compression calculation
        compression_ratio = round((1 - count/T) * 100, 1) if T > 0 else 0
        frames_kept = round((count/T) * 100, 1) if T > 0 else 0
        
        print(f"DEBUG: compression_ratio={compression_ratio}, frames_kept={frames_kept}")
        
        # Additional debug: Check summary array
        unique_summary = np.unique(summary)
        selected_frames = np.sum(summary == 1)
        print(f"DEBUG: Unique values in summary: {unique_summary}")
        print(f"DEBUG: Frames marked as 1: {selected_frames}")
        print(f"DEBUG: Total summary length: {len(summary)}")
        
        stats = {
            'original_frames': T,
            'summary_frames': count,
            'selected_frames': selected_frames,  # Add this for verification
            'compression_ratio': compression_ratio,
            'summary_duration': round(summary_duration, 1),
            'original_duration_est': round(original_duration, 1),
            'frames_kept': frames_kept,
            'summary_ratio': round(count/T, 3) if T > 0 else 0,  # Add ratio for debug
            'visualization': viz_fig,  # Add visualization chart
            'probs': probs,  # Add for additional analysis
            'summary': summary  # Add summary array
        }
        
        if progress_callback:
            progress_callback(100)
        
        # Cleanup temporary directory
        shutil.rmtree(tmp)
        
        return True, final_output_path, stats, None
        
    except Exception as e:
        # Cleanup on error
        if 'tmp' in locals():
            shutil.rmtree(tmp, ignore_errors=True)
        return False, None, None, f"Unexpected error: {str(e)}"

def create_video_analysis_visualization(probs, summary, temporal_analysis=None):
    """Create comprehensive video analysis visualization"""
      # Create frame indices
    frame_indices = np.arange(len(summary))
      # Create subplots - only 2 charts (remove temporal analysis)
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            '🎯 Frame Selection Overview',
            '📊 Frame Importance Scores'
        ),
        vertical_spacing=0.25,
        row_heights=[0.5, 0.5]
    )
    
    # Update subplot title annotations to center them
    for annotation in fig['layout']['annotations']:
        annotation['x'] = 0.5  # Center horizontally
        annotation['xanchor'] = 'center'  # Anchor from center
        annotation['font'] = dict(size=16, color='#ffffff', family='Inter')  # Style the titles
    
    # 1. Frame Selection Visualization (existing)
    fig.add_trace(
        go.Bar(
            x=frame_indices,
            y=summary,
            marker_color=['#48bb78' if summary[i] == 1 else '#2d3748' for i in range(len(summary))],
            name='Frame Selection',
            text=['✓' if summary[i] == 1 else '' for i in range(len(summary))],
            textposition='inside',
            hovertemplate='<b>Frame %{x}</b><br>Selected: %{customdata}<br><extra></extra>',
            customdata=['Yes' if summary[i] == 1 else 'No' for i in range(len(summary))],
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Frame Importance Scores
    fig.add_trace(
        go.Scatter(
            x=frame_indices,
            y=probs,
            mode='lines+markers',
            line=dict(color='#667eea', width=2),
            marker=dict(size=4, color='#667eea'),
            name='Importance Score',
            hovertemplate='<b>Frame %{x}</b><br>Score: %{y:.3f}<br><extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Highlight selected frames in score chart
    selected_indices = frame_indices[summary == 1]
    selected_scores = probs[summary == 1]
    
    fig.add_trace(
        go.Scatter(
            x=selected_indices,
            y=selected_scores,
            mode='markers',
            marker=dict(size=8, color='#48bb78', symbol='diamond'),
            name='Selected Frames',
            hovertemplate='<b>Frame %{x}</b><br>Score: %{y:.3f}<br>Status: Selected<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )        # Update layout with dark theme
    fig.update_layout(
        title={
            'text': '📈 Video Analysis Dashboard',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#667eea', 'family': 'Inter'}
        },
        height=600,  # Fixed height for 2 subplots
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(15,15,15,0.9)',
        font={'color': '#ffffff', 'family': 'Inter'},
        margin=dict(l=60, r=60, t=100, b=60)
    )
    
    # Update axes styling for all subplots (2 subplots)
    for i in range(1, 3):  # Fixed to 2 subplots
        fig.update_xaxes(
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)',
            linecolor='rgba(255,255,255,0.3)',
            title_font=dict(size=12, color='#a0a0a0'),
            row=i, col=1
        )
        fig.update_yaxes(
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)',
            linecolor='rgba(255,255,255,0.3)',
            title_font=dict(size=12, color='#a0a0a0'),
            row=i, col=1
        )
    
    # Specific axis labels
    fig.update_xaxes(title_text="Frame Index", row=1, col=1)
    fig.update_yaxes(title_text="Selected", range=[-0.1, 1.1], row=1, col=1)
    
    fig.update_xaxes(title_text="Frame Index", row=2, col=1)
    fig.update_yaxes(title_text="Importance Score", row=2, col=1)
    
    return fig


# Configure the page
st.set_page_config(
    page_title="AI Video Summarization",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for beautiful modern design with dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Theme - Dark Modern */
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .subtitle {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    
    /* Section Headers */
    .section-header {
        color: #667eea;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: left;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    
    /* Cards - Minimalist Dark */
    .info-card {
        background: rgba(25, 25, 25, 0.9);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Results Card - Success State */
    .results-card {
        background: rgba(25, 45, 30, 0.9);
        border: 1px solid #48bb78;
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 20px rgba(72, 187, 120, 0.2);
    }
    
    /* Waiting State Card */
    .waiting-card {
        background: rgba(25, 25, 25, 0.9);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .waiting-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    
    .waiting-title {
        color: #667eea;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .waiting-subtitle {
        color: #a0a0a0;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Process Steps - Clean Design */
    .process-steps {
        background: rgba(30, 30, 30, 0.8);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .steps-title {
        color: #667eea;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .step-item {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        padding: 0.5rem;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    
    .step-item:hover {
        background: rgba(102, 126, 234, 0.1);
    }
    
    .step-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.7rem;
        margin-right: 0.8rem;
        flex-shrink: 0;
    }
    
    .step-text {
        color: #ffffff;
        font-weight: 400;
        font-size: 0.9rem;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    /* Stats - Clean Minimal */
    .stat-item {
        background: rgba(30, 30, 30, 0.9);
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
        margin: 0.5rem 0;
    }
    
    .stat-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        border-color: #667eea;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.3rem;
        display: block;
    }
    
    .stat-label {
        color: #a0a0a0;
        font-weight: 500;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Progress Bar */
    .progress-container {
        background: rgba(25, 25, 25, 0.9);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .progress-text {
        font-weight: 600;
        color: #667eea;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Sidebar Dark Theme */
    .css-1d391kg {
        background: rgba(15, 15, 15, 0.98);
        backdrop-filter: blur(10px);
    }
    
    /* Fix text colors in dark theme */
    .stMarkdown p, .stText, h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    
    /* Selectbox styling */
    .stSelectbox label {
        color: #ffffff !important;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(30, 30, 30, 0.9) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: #ffffff !important;
    }
    
    /* File uploader styling */
    .stFileUploader label {
        color: #ffffff !important;
    }
    
    .stFileUploader > div {
        background-color: rgba(25, 25, 25, 0.9) !important;
        border: 2px dashed rgba(102, 126, 234, 0.5) !important;
        border-radius: 8px !important;
    }
    
    /* Buttons - Modern Minimal */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.8rem 2rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Download Button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.8rem 2rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3) !important;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(72, 187, 120, 0.4) !important;
    }
    
    /* Footer - Clean Minimal */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: rgba(20, 20, 20, 0.9);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .footer h3 {
        color: #667eea;
        margin-bottom: 0.5rem;
        font-size: 1.3rem;
    }
      .footer p {
        color: #a0a0a0;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }    /* Enhanced chart container styling */
    .stPlotlyChart {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Plotly chart styling */
    .plotly {
        background: rgba(25, 25, 25, 0.9) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    /* Video player styling */
    .stVideo {
        border-radius: 12px !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
        overflow: hidden !important;
    }
    
    .stVideo > div {
        border-radius: 12px !important;
    }
    
    .stVideo video {
        border-radius: 12px !important;
        width: 100% !important;
        height: auto !important;
    }
    
    /* Video container enhancement */
    .video-container {
        background: rgba(25, 25, 25, 0.9);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin: 1rem 0;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 30, 30, 0.5);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

# Main header with modern design
st.markdown("# 🎬 AI Video Summarization")

# Initialize session state variables
if 'results' not in st.session_state:
    st.session_state.results = None
if 'current_video_name' not in st.session_state:
    st.session_state.current_video_name = None
if 'processing' not in st.session_state:
    st.session_state.processing = False


# Check FFmpeg availability
ffmpeg_available = check_ffmpeg()
if not ffmpeg_available:
    st.error("⚠️ **FFmpeg not found!** Please install FFmpeg for video processing.")
    st.markdown("**Download:** https://ffmpeg.org/download.html")
    st.stop()

# Get available models
available_models = get_available_models()

# Sidebar configuration with beautiful model selection
st.sidebar.markdown("## 🤖 Model Configuration")

if not available_models:
    st.sidebar.error("❌ No models found in log directory!")
    st.error("**No models found!** Please ensure model checkpoints exist in the log directory.")
    st.stop()

# Model selection with hierarchy
st.sidebar.markdown("### 🔬 Select Model Architecture")
model_types = list(available_models.keys())
model_descriptions = {
    'DR-DSN': 'DR-DSN',
    'DR-DSNsup': 'DR-DSNsup',
    'D-DSN': 'D-DSN',
    'D-DSN-nolambda': 'D-DSN-nolambda',
    'DSNsup': 'DSNsup',
    'R-DSN': 'R-DSN'
}

selected_model_type = st.sidebar.selectbox(
    "Model Type:",
    model_types,
    format_func=lambda x: model_descriptions.get(x, x),
    help="Different model architectures with varying performance characteristics"
)

# Dataset selection
st.sidebar.markdown("### 📊 Select Training Dataset")
datasets = list(available_models[selected_model_type].keys())
dataset_descriptions = {
    'tvsum': '📺 TV Sum Dataset (50 videos, diverse content)',
    'summe': '🎬 SumMe Dataset (25 videos, user summaries)'
}

selected_dataset = st.sidebar.selectbox(
    "Dataset:",
    datasets,
    format_func=lambda x: dataset_descriptions.get(x, x),
    help="Dataset used for model training affects summarization style"
)

# Split selection
st.sidebar.markdown("### 🎯 Select Cross-Validation Split")
splits = available_models[selected_model_type][selected_dataset]

selected_split_idx = st.sidebar.selectbox(
    "Split:",
    range(len(splits)),
    format_func=lambda x: f"Split {x} ({splits[x]['split']})",
    help="Different cross-validation splits for robust evaluation"
)

selected_split = splits[selected_split_idx]
model_path = selected_split['path']

# Show current selection
st.sidebar.markdown("---")
st.sidebar.markdown("### ✅ Current Selection")
st.sidebar.markdown(f"""
**🔬 Model:** {selected_model_type}  
**📊 Dataset:** {selected_dataset}  
**🎯 Split:** {selected_split['split']}  
**📂 Directory:** `{selected_split['dir']}`
""")

# Verify model exists
if not osp.exists(model_path):
    st.sidebar.error("❌ Selected model checkpoint not found!")
    st.error(f"**Model not found:** `{model_path}`")
    st.stop()
else:
    st.sidebar.success("✅ Model checkpoint verified")

# Output settings
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Output Settings")
fps_out = st.sidebar.slider("Output FPS", min_value=5, max_value=60, value=5, step=5, help="Frames per second for the output video")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Summary Generation Settings")

# Summary method - only knapsack (as per original paper)
summary_method = "knapsack"

# Summary proportion
summary_proportion = st.sidebar.slider(
    "Summary Length (%)",
    min_value=5,
    max_value=50,
    value=15,
    step=5,
    help="Percentage of original video length for the summary (typically 15% as mentioned in paper)"
)

# Device info
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.markdown("---")
st.sidebar.markdown("### 💻 System Information")
if device == 'cuda':
    st.sidebar.success(f"🚀 GPU: {torch.cuda.get_device_name()}")
else:
    st.sidebar.info("💻 Using CPU")

# About section
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
**🔬 Features:**
- GoogLeNet pool5 features (1024-dim)
- Knapsack optimization algorithm
- Multiple model architectures
- Cross-validation splits

**📁 Supported Formats:**
MP4, AVI, MOV, MKV, MPEG4

**⚡ Output:**
Web-compatible MP4 with H.264
""")

# Main content with vertical layout
st.markdown("## 📁 Upload Video")

uploaded_file = st.file_uploader(
    "Choose your video file",
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Upload a video file to generate an AI-powered summary",
    label_visibility="collapsed"
)

# Check if a new video is uploaded (different from current one)
if uploaded_file is not None:
    if st.session_state.current_video_name != uploaded_file.name:
        # New video uploaded, reset results
        st.session_state.results = None

        st.session_state.current_video_name = uploaded_file.name
        st.session_state.processing = False

if uploaded_file is not None:
    # Add spacing between file uploader and video information
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display video info in a clean card
    with st.container():
        st.markdown("### 📄 Video Information")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown(f"**📁 Name:** {uploaded_file.name}")
            st.markdown(f"**💾 Size:** {uploaded_file.size / (1024*1024):.2f} MB")
        with col_info2:
            st.markdown(f"**📊 Type:** {uploaded_file.type}")
            st.markdown(f"**🤖 Model:** {selected_model_type}")
    
    # Original Video Preview Section
    st.markdown("---")
    st.markdown("### 🎬 Original Video Preview")
    
    with st.container():
        # Add custom CSS to control video size
        st.markdown("""
        <style>
            .stVideo {
                width: 60% !important;  /* Adjust percentage as needed */
                margin: 0 auto;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Show original video
        st.video(uploaded_file.getvalue())


# Processing Center Section
st.markdown("---")
st.markdown("## 🎯 Processing Center")
if uploaded_file is not None:
    # Check if we have saved results for this video
    if st.session_state.results is not None:
        # Display saved results
        st.success("### ✅ Summary Generated Successfully!")
        
        # Display Frame Selection Visualization chart directly
        st.markdown("#### Frame Selection Analysis")
          # Display the visualization chart
        if 'visualization' in st.session_state.results and st.session_state.results['visualization'] is not None:
            st.plotly_chart(st.session_state.results['visualization'], use_container_width=True)
        else:
            st.warning("Visualization data not available")
          
        # Key metrics in a clean layout
        st.markdown("#### Summary Statistics")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        stats = st.session_state.results
        with col_stat1:
            st.metric(
                label="Original Frames",
                value=f"{stats['original_frames']:,}"
            )
        
        with col_stat2:
            st.metric(
                label="Summary Frames", 
                value=f"{stats['summary_frames']:,}"
            )
        
        with col_stat3:
            st.metric(
                label="Compression Ratio",
                value=f"{stats['compression_ratio']:.1f}%"
            )
        
        with col_stat4:
            st.metric(
                label="Summary Duration",
                value=f"{stats['summary_duration']:.1f}s"
            )
          # Duration info in clean format
        st.info(f"""
        **Original Duration:** {stats['original_duration_est']:.1f}s → **Summary:** {stats['summary_duration']:.1f}s  
        **Selected:** {stats['selected_frames']}/{stats['original_frames']} frames (Ratio: {stats['summary_ratio']:.3f})
        """)
        
        # # Video Comparison Section
        # st.markdown("---")
        # st.markdown("#### 🎭 Video Comparison")
        
        # if 'output_path' in st.session_state.results and osp.exists(st.session_state.results['output_path']):
        #     # Read summary video
        #     with open(st.session_state.results['output_path'], 'rb') as video_file:
        #         summary_video_bytes = video_file.read()
            
        #     # Side-by-side comparison
        #     col_orig, col_summary = st.columns(2)
            
        #     with col_orig:
        #         st.markdown("##### 📹 Original Video")
        #         with st.container():
        #             st.markdown('<div class="video-container">', unsafe_allow_html=True)
        #             st.video(uploaded_file.getvalue())
                    
        #             # Original video metrics
        #             st.metric("📊 File Size", f"{uploaded_file.size / (1024*1024):.1f} MB")
        #             st.metric("⏱️ Estimated Duration", f"{stats['original_duration_est']:.1f}s")
        #             st.metric("🎬 Total Frames", f"{stats['original_frames']:,}")
                    
        #             st.markdown('</div>', unsafe_allow_html=True)
            
        #     with col_summary:
        #         st.markdown("##### ✨ AI Summary")
        #         with st.container():
        #             st.markdown('<div class="video-container">', unsafe_allow_html=True)
        #             st.video(summary_video_bytes)
                    
        #             # Summary video metrics
        #             summary_size = len(summary_video_bytes) / (1024*1024)
        #             st.metric("📊 File Size", f"{summary_size:.1f} MB")
        #             st.metric("⏱️ Duration", f"{stats['summary_duration']:.1f}s")
        #             st.metric("🎬 Selected Frames", f"{stats['summary_frames']:,}")
                    
        #             st.markdown('</div>', unsafe_allow_html=True)
            
        #     # Comparison metrics
        #     st.markdown("##### 📊 Comparison Metrics")
        #     col_comp1, col_comp2, col_comp3, col_comp4 = st.columns(4)
            
        #     with col_comp1:
        #         size_reduction = (1 - (len(summary_video_bytes) / uploaded_file.size)) * 100
        #         st.metric("🗜️ Size Reduction", f"{size_reduction:.1f}%")
            
        #     with col_comp2:
        #         time_reduction = (1 - (stats['summary_duration'] / stats['original_duration_est'])) * 100
        #         st.metric("⏱️ Time Reduction", f"{time_reduction:.1f}%")
            
        #     with col_comp3:
        #         st.metric("🎯 Compression Ratio", f"{stats['compression_ratio']:.1f}%")
            
        #     with col_comp4:
        #         st.metric("📐 Selection Ratio", f"{stats['summary_ratio']:.3f}")
          # Detailed Video Player Section
        if 'output_path' in st.session_state.results and osp.exists(st.session_state.results['output_path']):
            st.markdown("---")
            st.markdown("#### 🎬 Detailed Summary Video Player")
            
            # Read video file for both display and download (if not already read)
            if 'summary_video_bytes' not in locals():
                with open(st.session_state.results['output_path'], 'rb') as video_file:
                    summary_video_bytes = video_file.read()
            
            # Create a styled container for the detailed video player
            with st.container():
                
                # Display video player using st.video with larger size
                st.video(summary_video_bytes)
            
            
            st.markdown("---")
            st.markdown("#### 📥 Download")
            
            # Simple download button (no columns)
            base_name = osp.splitext(uploaded_file.name)[0]
            st.download_button(
                label="📥 Download Summary Video",
                data=summary_video_bytes,
                file_name=f'{base_name}_summary.mp4',
                mime='video/mp4',
                use_container_width=True,  # Not using full width for better aesthetics
                help="Download the AI-generated video summary"
            )
            
            # Compression status info
            if stats['compression_ratio'] > 0:
                st.success(f"✨ **Summary ready!** Video compressed by {stats['compression_ratio']:.1f}% - Duration reduced from {stats['original_duration_est']:.1f}s to {stats['summary_duration']:.1f}s")
            else:
                st.warning(f"⚠️ **Warning:** Model selected all frames ({stats['summary_frames']}/{stats['original_frames']}). This suggests the model may need adjustment.")
            
        # Add button to generate new summary
        st.markdown("---")
        if st.button("🔄 Generate New Summary", use_container_width=True):
            st.session_state.results = None
            st.rerun()
            
    else:
        # Show processing interface
        # Beautiful processing button
        process_button = st.button(
            "🚀 Generate AI Summary",
            type="primary",
            use_container_width=True,
            help="Start the AI-powered video summarization process"
        )
        
        if process_button:
            st.session_state.processing = True
            
        if st.session_state.processing:
            # Initialize progress tracking with clean container
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                video_path = tmp_file.name
            
            try:
                # Create progress and status callbacks
                def update_progress(value):
                    progress_bar.progress(value)
                
                def update_status(message):
                    status_text.markdown(f'⚡ {message}')
                
                # Run summarization
                success, output_path, stats, error_msg = summarize_video(
                    video_path, model_path, fps_out, update_progress, update_status
                )
                
                # Clean up temporary file
                os.unlink(video_path)
                
                if success:
                    # Save results to session state
                    st.session_state.results = stats.copy()
                    st.session_state.results['output_path'] = output_path
                    st.session_state.processing = False
                    st.rerun()
                
                else:
                    # Show error message
                    st.error(f"❌ **Processing Failed:** {error_msg}")
                    st.markdown("""
                    **💡 Troubleshooting Tips:**
                    - Ensure video file is not corrupted
                    - Check if FFmpeg is properly installed  
                    - Try with a different video format
                    - Verify sufficient disk space
                    """)
                    st.session_state.processing = False
                    
            except Exception as e:
                # Clean up temporary file on exception
                if osp.exists(video_path):
                    os.unlink(video_path)
                st.error(f"❌ **Unexpected Error:** {str(e)}")
                st.session_state.processing = False
else:
    # Show waiting state - clean and professional using native Streamlit components
    with st.container():
        st.markdown("### 🤖 AI Ready for Processing")
        st.markdown("*Upload a video file to begin AI-powered summarization*")
        
        st.markdown("---")
        
        # Processing pipeline steps using native Streamlit
        st.markdown("#### 🎯 Processing Pipeline")
        
        st.markdown("**1.**    Extract frames at 2 FPS")
        st.markdown("**2.**    Generate GoogLeNet features.")
        st.markdown("**3.**    Apply Deep RL model.")
        st.markdown("**4.**    Optimize with knapsack algorithm.")
        st.markdown("**5.**    Create web-compatible summary.")


# Add some spacing before footer
st.markdown("---")

