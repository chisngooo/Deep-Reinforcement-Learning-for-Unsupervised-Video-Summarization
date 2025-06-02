"""
Video Summarization Pipeline Visualization
Vẽ pipeline chi tiết end-to-end cho hệ thống Deep Reinforcement Learning Video Summarization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle
import numpy as np

def create_pipeline_diagram():
    """Tạo biểu đồ pipeline chi tiết end-to-end"""
    
    # Thiết lập figure với kích thước lớn
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Định nghĩa màu sắc cho các giai đoạn
    colors = {
        'input': '#FF6B6B',      # Đỏ - Input
        'feature': '#4ECDC4',    # Xanh lá - Feature Extraction
        'segment': '#45B7D1',    # Xanh dương - Segmentation
        'model': '#96CEB4',      # Xanh nhạt - Model Processing
        'selection': '#FFEAA7',  # Vàng - Selection
        'output': '#DDA0DD',     # Tím - Output
        'diversity': '#FFB347'   # Cam - Diversity Enhancement
    }
    
    # Title
    ax.text(5, 11.5, 'VIDEO SUMMARIZATION PIPELINE - END TO END', 
            fontsize=24, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    # ==================== STAGE 1: INPUT PROCESSING ====================
    stage1_y = 10.5
    
    # Input Video
    input_box = FancyBboxPatch((0.2, stage1_y-0.3), 1.6, 0.6, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], alpha=0.8)
    ax.add_patch(input_box)
    ax.text(1, stage1_y, 'INPUT VIDEO\n(.mp4/.avi)', fontsize=10, fontweight='bold', 
            ha='center', va='center')
    
    # Arrow to preprocessing
    ax.annotate('', xy=(2.2, stage1_y), xytext=(1.8, stage1_y),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Video Preprocessing
    preprocess_box = FancyBboxPatch((2.2, stage1_y-0.3), 1.8, 0.6,
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['input'], alpha=0.8)
    ax.add_patch(preprocess_box)
    ax.text(3.1, stage1_y, 'VIDEO PREPROCESSING\n• Extract frames (2 FPS)\n• Resize to 224x224', 
            fontsize=9, ha='center', va='center')
    
    # ==================== STAGE 2: FEATURE EXTRACTION ====================
    stage2_y = 9.2
    
    # Arrow down
    ax.annotate('', xy=(3.1, stage2_y+0.3), xytext=(3.1, stage1_y-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Feature Extraction
    feature_box = FancyBboxPatch((2.2, stage2_y-0.3), 1.8, 0.6,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['feature'], alpha=0.8)
    ax.add_patch(feature_box)
    ax.text(3.1, stage2_y, 'FEATURE EXTRACTION\n• GoogleNet Pool5\n• Shape: (T, 1024)', 
            fontsize=9, ha='center', va='center')
    
    # ==================== STAGE 3: TEMPORAL SEGMENTATION ====================
    stage3_y = 7.9
    
    # Arrow down
    ax.annotate('', xy=(3.1, stage3_y+0.3), xytext=(3.1, stage2_y-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # KTS Segmentation
    kts_box = FancyBboxPatch((1.5, stage3_y-0.4), 3.2, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['segment'], alpha=0.8)
    ax.add_patch(kts_box)
    ax.text(3.1, stage3_y, 'TEMPORAL SEGMENTATION (KTS)\n• Kernel Temporal Segmentation\n• Find shot boundaries\n• Output: Change points array', 
            fontsize=9, ha='center', va='center')
    
    # KTS Details (side box)
    kts_detail_box = FancyBboxPatch((5.2, stage3_y-0.5), 2.3, 1.0,
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['segment'], alpha=0.5)
    ax.add_patch(kts_detail_box)
    ax.text(6.35, stage3_y, 'KTS ALGORITHM:\n1. Compute kernel matrix\n2. Dynamic programming\n3. Find optimal change points\n4. Create segments', 
            fontsize=8, ha='center', va='center')
    
    # Arrow to KTS details
    ax.annotate('', xy=(5.2, stage3_y), xytext=(4.7, stage3_y),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # ==================== STAGE 4: MODEL PROCESSING ====================
    stage4_y = 6.4
    
    # Arrow down
    ax.annotate('', xy=(3.1, stage4_y+0.4), xytext=(3.1, stage3_y-0.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # DSN Model
    model_box = FancyBboxPatch((1.8, stage4_y-0.4), 2.6, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['model'], alpha=0.8)
    ax.add_patch(model_box)
    ax.text(3.1, stage4_y, 'DSN MODEL PROCESSING\n• Deep Summarization Network\n• Predict frame importance\n• Output: Probability scores', 
            fontsize=9, ha='center', va='center')
    
    # Model Architecture Details
    arch_box = FancyBboxPatch((5.2, stage4_y-0.6), 2.5, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['model'], alpha=0.5)
    ax.add_patch(arch_box)
    ax.text(6.45, stage4_y, 'DSN ARCHITECTURE:\n• Bi-LSTM layers\n• Self-attention\n• Fully connected\n• Sigmoid activation\n• Reinforcement learning', 
            fontsize=8, ha='center', va='center')
    
    # Arrow to model details
    ax.annotate('', xy=(5.2, stage4_y), xytext=(4.4, stage4_y),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # ==================== STAGE 5: SUMMARY GENERATION ====================
    stage5_y = 4.8
    
    # Arrow down
    ax.annotate('', xy=(3.1, stage5_y+0.5), xytext=(3.1, stage4_y-0.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Summary Generation Hub
    summary_box = FancyBboxPatch((2.0, stage5_y-0.5), 2.2, 1.0,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['selection'], alpha=0.8)
    ax.add_patch(summary_box)
    ax.text(3.1, stage5_y, 'SUMMARY GENERATION\n• Segment scoring\n• Selection algorithm\n• Length constraint (15%)', 
            fontsize=9, fontweight='bold', ha='center', va='center')
    
    # ==================== SELECTION METHODS ====================
    methods_y = 3.2
    
    # Method 1: Knapsack
    knap_box = FancyBboxPatch((0.2, methods_y-0.3), 1.5, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['selection'], alpha=0.8)
    ax.add_patch(knap_box)
    ax.text(0.95, methods_y, 'KNAPSACK\n(Basic)', fontsize=9, ha='center', va='center')
    
    # Method 2: Temporal Knapsack
    temp_knap_box = FancyBboxPatch((2.0, methods_y-0.3), 1.5, 0.6,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['diversity'], alpha=0.8)
    ax.add_patch(temp_knap_box)
    ax.text(2.75, methods_y, 'TEMPORAL\nKNAPSACK', fontsize=9, ha='center', va='center')
    
    # Method 3: Sliding Window
    slide_box = FancyBboxPatch((3.8, methods_y-0.3), 1.5, 0.6,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['diversity'], alpha=0.8)
    ax.add_patch(slide_box)
    ax.text(4.55, methods_y, 'SLIDING\nWINDOW', fontsize=9, ha='center', va='center')
    
    # Method 4: Uniform Sampling
    uniform_box = FancyBboxPatch((5.6, methods_y-0.3), 1.5, 0.6,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['diversity'], alpha=0.8)
    ax.add_patch(uniform_box)
    ax.text(6.35, methods_y, 'UNIFORM\nSAMPLING', fontsize=9, ha='center', va='center')
    
    # Method 5: Hybrid
    hybrid_box = FancyBboxPatch((7.4, methods_y-0.3), 1.5, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['diversity'], alpha=0.8)
    ax.add_patch(hybrid_box)
    ax.text(8.15, methods_y, 'HYBRID\n(Best of all)', fontsize=9, ha='center', va='center')
    
    # Arrows from summary to methods
    for x_pos in [0.95, 2.75, 4.55, 6.35, 8.15]:
        ax.annotate('', xy=(x_pos, methods_y+0.3), xytext=(3.1, stage5_y-0.5),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # ==================== TEMPORAL DIVERSITY ENHANCEMENT ====================
    diversity_y = 1.8
    
    # Diversity Enhancement Box
    div_box = FancyBboxPatch((1.5, diversity_y-0.4), 6.0, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['diversity'], alpha=0.8)
    ax.add_patch(div_box)
    ax.text(4.5, diversity_y, 'TEMPORAL DIVERSITY ENHANCEMENT\n• Calculate diversity penalty • Uniform distribution analysis • Bias detection (beginning/end/balanced)', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    
    # Arrows from methods to diversity
    for x_pos in [2.75, 4.55, 6.35, 8.15]:
        ax.annotate('', xy=(4.5, diversity_y+0.4), xytext=(x_pos, methods_y-0.3),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='orange'))
    
    # ==================== FINAL OUTPUT ====================
    output_y = 0.5
    
    # Arrow down
    ax.annotate('', xy=(4.5, output_y+0.3), xytext=(4.5, diversity_y-0.4),
                arrowprops=dict(arrowstyle='->', lw=3, color='purple'))
    
    # Final Output
    output_box = FancyBboxPatch((3.2, output_y-0.2), 2.6, 0.4,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['output'], alpha=0.8)
    ax.add_patch(output_box)
    ax.text(4.5, output_y, 'FINAL VIDEO SUMMARY\n(Binary keyshot vector)', 
            fontsize=11, fontweight='bold', ha='center', va='center')
    
    # ==================== SIDE INFORMATION BOXES ====================
    
    # Parameters Box
    param_box = FancyBboxPatch((8.2, 10.0), 1.6, 1.8,
                               boxstyle="round,pad=0.1",
                               facecolor='lightblue', alpha=0.7)
    ax.add_patch(param_box)
    ax.text(9.0, 10.9, 'KEY PARAMETERS:\n\n• FPS: 2\n• Proportion: 15%\n• Max segments: 100\n• Diversity weight: 0.2\n• Kernel: Cosine', 
            fontsize=8, ha='center', va='center')
    
    # Evaluation Box
    eval_box = FancyBboxPatch((8.2, 7.8), 1.6, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor='lightgreen', alpha=0.7)
    ax.add_patch(eval_box)
    ax.text(9.0, 8.55, 'EVALUATION:\n\n• F1-Score\n• Precision/Recall\n• Temporal analysis\n• User study', 
            fontsize=8, ha='center', va='center')
    
    # Datasets Box
    data_box = FancyBboxPatch((8.2, 5.5), 1.6, 1.8,
                              boxstyle="round,pad=0.1",
                              facecolor='lightyellow', alpha=0.7)
    ax.add_patch(data_box)
    ax.text(9.0, 6.4, 'DATASETS:\n\n• TVSum\n• SumMe\n• OVP\n• YouTube\n\n(H5 format)', 
            fontsize=8, ha='center', va='center')
    
    # ==================== LEGEND ====================
    legend_elements = [
        mpatches.Patch(color=colors['input'], label='Input Processing'),
        mpatches.Patch(color=colors['feature'], label='Feature Extraction'),
        mpatches.Patch(color=colors['segment'], label='Segmentation'),
        mpatches.Patch(color=colors['model'], label='Model Processing'),
        mpatches.Patch(color=colors['selection'], label='Basic Selection'),
        mpatches.Patch(color=colors['diversity'], label='Diversity Enhancement'),
        mpatches.Patch(color=colors['output'], label='Final Output')
    ]
    
    ax.legend(handles=legend_elements, loc='lower left', 
              bbox_to_anchor=(0, 0), fontsize=10)
    
    # ==================== FLOW INDICATORS ====================
    
    # Main flow arrow (left side)
    ax.annotate('MAIN FLOW', xy=(0.1, 6), xytext=(0.1, 8),
                arrowprops=dict(arrowstyle='<->', lw=3, color='red'),
                fontsize=12, fontweight='bold', ha='center', rotation=90)
    
    # Enhancement flow arrow (right side)
    ax.annotate('ENHANCEMENT', xy=(9.8, 2), xytext=(9.8, 4),
                arrowprops=dict(arrowstyle='<->', lw=3, color='orange'),
                fontsize=12, fontweight='bold', ha='center', rotation=90)
    
    plt.tight_layout()
    return fig

def create_detailed_algorithm_flow():
    """Tạo biểu đồ chi tiết về flow của các algorithms"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # ==================== KTS ALGORITHM ====================
    ax1.set_title('KTS (Kernel Temporal Segmentation)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # KTS Steps
    steps_kts = [
        ('Input Features\n(T, 1024)', 9),
        ('Compute Kernel Matrix\n(Cosine Similarity)', 7.5),
        ('Dynamic Programming\nOptimization', 6),
        ('Find Change Points\n(Shot Boundaries)', 4.5),
        ('Create Segments\n[start, end] pairs', 3)
    ]
    
    for i, (step, y) in enumerate(steps_kts):
        box = FancyBboxPatch((2, y-0.4), 6, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor='lightblue', alpha=0.8)
        ax1.add_patch(box)
        ax1.text(5, y, step, fontsize=10, ha='center', va='center')
        
        if i < len(steps_kts) - 1:
            ax1.annotate('', xy=(5, y-0.6), xytext=(5, y-0.4),
                        arrowprops=dict(arrowstyle='->', lw=2))
    
    # ==================== DSN MODEL ====================
    ax2.set_title('DSN (Deep Summarization Network)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # DSN Architecture
    layers_dsn = [
        ('Features (T, 1024)', 9),
        ('Bi-LSTM\n(Hidden: 256)', 7.5),
        ('Self-Attention\nMechanism', 6),
        ('Fully Connected\n(256 → 1)', 4.5),
        ('Sigmoid Activation\nP(select) ∈ [0,1]', 3)
    ]
    
    for i, (layer, y) in enumerate(layers_dsn):
        box = FancyBboxPatch((2, y-0.4), 6, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor='lightgreen', alpha=0.8)
        ax2.add_patch(box)
        ax2.text(5, y, layer, fontsize=10, ha='center', va='center')
        
        if i < len(layers_dsn) - 1:
            ax2.annotate('', xy=(5, y-0.6), xytext=(5, y-0.4),
                        arrowprops=dict(arrowstyle='->', lw=2))
    
    # ==================== KNAPSACK VARIANTS ====================
    ax3.set_title('Selection Algorithms Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 12)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    # Basic Knapsack
    knap_box = FancyBboxPatch((0.5, 7), 2.5, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor='yellow', alpha=0.8)
    ax3.add_patch(knap_box)
    ax3.text(1.75, 7.75, 'BASIC KNAPSACK\n\n• Max total score\n• Length constraint\n• No diversity', 
             fontsize=9, ha='center', va='center')
    
    # Temporal Knapsack
    temp_box = FancyBboxPatch((3.5, 7), 2.5, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor='orange', alpha=0.8)
    ax3.add_patch(temp_box)
    ax3.text(4.75, 7.75, 'TEMPORAL KNAPSACK\n\n• Adjusted scores\n• Position penalty\n• Better distribution', 
             fontsize=9, ha='center', va='center')
    
    # Sliding Window
    slide_box = FancyBboxPatch((6.5, 7), 2.5, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor='lightcoral', alpha=0.8)
    ax3.add_patch(slide_box)
    ax3.text(7.75, 7.75, 'SLIDING WINDOW\n\n• Divide into windows\n• Best from each\n• Guaranteed spread', 
             fontsize=9, ha='center', va='center')
    
    # Uniform Sampling
    uniform_box = FancyBboxPatch((0.5, 4.5), 2.5, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor='lightblue', alpha=0.8)
    ax3.add_patch(uniform_box)
    ax3.text(1.75, 5.25, 'UNIFORM SAMPLING\n\n• Fixed intervals\n• Perfect distribution\n• May miss peaks', 
             fontsize=9, ha='center', va='center')
    
    # Hybrid
    hybrid_box = FancyBboxPatch((3.5, 4.5), 2.5, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor='plum', alpha=0.8)
    ax3.add_patch(hybrid_box)
    ax3.text(4.75, 5.25, 'HYBRID METHOD\n\n• Try all methods\n• Compare results\n• Best overall score', 
             fontsize=9, ha='center', va='center')
    
    # Comparison Result
    result_box = FancyBboxPatch((6.5, 4.5), 2.5, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor='lightgreen', alpha=0.8)
    ax3.add_patch(result_box)
    ax3.text(7.75, 5.25, 'SELECTION RESULT\n\n• Optimal segments\n• Temporal balance\n• Quality maximized', 
             fontsize=9, ha='center', va='center')
    
    # Arrows showing flow
    arrows = [(1.75, 6.8), (4.75, 6.8), (7.75, 6.8), (1.75, 4.3), (4.75, 4.3)]
    for x, y in arrows:
        ax3.annotate('', xy=(7.75, 4.3), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # ==================== TEMPORAL DIVERSITY ANALYSIS ====================
    ax4.set_title('Temporal Diversity Analysis', fontsize=14, fontweight='bold')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    
    # Create sample timeline
    timeline_y = 8
    ax4.plot([1, 9], [timeline_y, timeline_y], 'k-', lw=3)
    ax4.text(5, timeline_y + 0.5, 'Video Timeline', fontsize=12, ha='center', fontweight='bold')
    
    # Sample segments - Bad distribution
    bad_segments = [1.5, 2, 2.5, 3, 3.5]  # Clustered at beginning
    for i, seg in enumerate(bad_segments):
        ax4.plot(seg, timeline_y, 'ro', markersize=8)
        if i == 0:
            ax4.text(seg, timeline_y - 0.3, 'Selected\nSegments', fontsize=8, ha='center')
    
    ax4.text(5, timeline_y - 1, 'BAD: Clustered at beginning', fontsize=11, ha='center', 
             color='red', fontweight='bold')
    
    # Sample segments - Good distribution
    timeline_y2 = 5.5
    ax4.plot([1, 9], [timeline_y2, timeline_y2], 'k-', lw=3)
    
    good_segments = [1.5, 3.2, 5, 6.8, 8.5]  # Well distributed
    for seg in good_segments:
        ax4.plot(seg, timeline_y2, 'go', markersize=8)
    
    ax4.text(5, timeline_y2 - 1, 'GOOD: Well distributed', fontsize=11, ha='center', 
             color='green', fontweight='bold')
    
    # Analysis metrics
    metrics_box = FancyBboxPatch((1, 2), 8, 2,
                                 boxstyle="round,pad=0.1",
                                 facecolor='lightyellow', alpha=0.8)
    ax4.add_patch(metrics_box)
    ax4.text(5, 3, 'DIVERSITY METRICS:\n\n' +
             '• Mean Position: 0.5 (balanced) vs 0.3 (beginning bias)\n' +
             '• Standard Deviation: >0.2 (distributed) vs <0.2 (concentrated)\n' +
             '• Range: >0.6 (good coverage) vs <0.6 (partial coverage)\n' +
             '• Gap Analysis: Uniform gaps vs Irregular gaps',
             fontsize=10, ha='center', va='center')
    
    plt.tight_layout()
    return fig

# Main execution
if __name__ == "__main__":
    print("🎬 Creating Video Summarization Pipeline Diagrams...")
    
    # Create main pipeline
    print("📊 Generating main pipeline diagram...")
    fig1 = create_pipeline_diagram()
    fig1.savefig('video_summarization_pipeline.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: video_summarization_pipeline.png")
    
    # Create detailed algorithm flow
    print("🔍 Generating detailed algorithm flow...")
    fig2 = create_detailed_algorithm_flow()
    fig2.savefig('algorithm_details.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: algorithm_details.png")
    
    print("\n🎉 Pipeline diagrams created successfully!")
    print("📁 Files generated:")
    print("   • video_summarization_pipeline.png - Main end-to-end pipeline")
    print("   • algorithm_details.png - Detailed algorithm breakdown")
    
    # Show the plots
    plt.show()
