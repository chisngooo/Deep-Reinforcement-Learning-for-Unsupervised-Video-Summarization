"""
Text-based Pipeline Visualization
Hiển thị pipeline chi tiết dưới dạng text để dễ đọc trong terminal/console
"""

def print_pipeline_overview():
    """In tổng quan pipeline end-to-end"""
    
    print("=" * 80)
    print("🎬 VIDEO SUMMARIZATION PIPELINE - END TO END OVERVIEW")
    print("=" * 80)
    
    pipeline_steps = [
        {
            "stage": "1. INPUT PROCESSING",
            "icon": "📥",
            "description": "Video input and preprocessing",
            "details": [
                "• Input: Raw video files (.mp4, .avi, etc.)",
                "• Extract frames at 2 FPS",
                "• Resize frames to 224x224 pixels",
                "• Convert to tensor format"
            ]
        },
        {
            "stage": "2. FEATURE EXTRACTION", 
            "icon": "🔍",
            "description": "Extract visual features from frames",
            "details": [
                "• Use pre-trained GoogleNet (Pool5 layer)",
                "• Extract 1024-dimensional features per frame",
                "• Output shape: (T, 1024) where T = number of frames",
                "• Features represent visual content"
            ]
        },
        {
            "stage": "3. TEMPORAL SEGMENTATION",
            "icon": "✂️", 
            "description": "Segment video into meaningful shots",
            "details": [
                "• Apply KTS (Kernel Temporal Segmentation)",
                "• Compute kernel matrix (cosine similarity)",
                "• Use dynamic programming to find optimal boundaries",
                "• Output: Array of change points [start, end] pairs"
            ]
        },
        {
            "stage": "4. DEEP MODEL PROCESSING",
            "icon": "🧠",
            "description": "Predict frame importance using DSN",
            "details": [
                "• Load trained DSN model (Deep Summarization Network)",
                "• Architecture: Bi-LSTM + Self-Attention + FC layers",
                "• Input: Feature sequence (T, 1024)",
                "• Output: Importance scores for each frame [0, 1]"
            ]
        },
        {
            "stage": "5. SEGMENT SCORING",
            "icon": "📊",
            "description": "Calculate importance score for each segment",
            "details": [
                "• Average frame scores within each segment",
                "• Create segment-level importance scores",
                "• Consider segment length (number of frames)",
                "• Prepare for selection algorithm"
            ]
        },
        {
            "stage": "6. SELECTION ALGORITHMS",
            "icon": "🎯",
            "description": "Select best segments for summary",
            "details": [
                "• Multiple algorithms available:",
                "  - Basic Knapsack (maximize score, length constraint)",
                "  - Temporal Knapsack (with position penalty)",
                "  - Sliding Window (guaranteed distribution)",
                "  - Uniform Sampling (fixed intervals)",
                "  - Hybrid (best of all methods)"
            ]
        },
        {
            "stage": "7. TEMPORAL DIVERSITY",
            "icon": "🌈",
            "description": "Enhance temporal distribution",
            "details": [
                "• Calculate diversity penalty for selection",
                "• Analyze distribution: concentrated/partial/distributed",
                "• Detect bias: beginning/end/balanced",
                "• Adjust selection to improve temporal spread"
            ]
        },
        {
            "stage": "8. FINAL OUTPUT",
            "icon": "🎉",
            "description": "Generate binary summary vector",
            "details": [
                "• Create binary keyshot vector (1=selected, 0=not selected)",
                "• Length constraint: typically 15% of original video",
                "• Output format compatible with evaluation metrics",
                "• Ready for video generation or evaluation"
            ]
        }
    ]
    
    for i, step in enumerate(pipeline_steps):
        print(f"\n{step['icon']} {step['stage']}")
        print(f"   📝 {step['description']}")
        for detail in step['details']:
            print(f"   {detail}")
        
        if i < len(pipeline_steps) - 1:
            print("   ⬇️  " + "─" * 40)

def print_algorithm_comparison():
    """So sánh các thuật toán selection"""
    
    print("\n" + "=" * 80)
    print("🔍 SELECTION ALGORITHMS COMPARISON")
    print("=" * 80)
    
    algorithms = [
        {
            "name": "BASIC KNAPSACK",
            "icon": "💼",
            "pros": [
                "✅ Maximizes total importance score",
                "✅ Respects length constraint",
                "✅ Fast computation O(n*W)",
                "✅ Proven optimal for score maximization"
            ],
            "cons": [
                "❌ No temporal diversity consideration",
                "❌ May cluster selections in high-score regions",
                "❌ Poor temporal distribution",
                "❌ Beginning/end bias possible"
            ],
            "use_case": "When only score maximization matters"
        },
        {
            "name": "TEMPORAL KNAPSACK", 
            "icon": "⏰",
            "pros": [
                "✅ Considers temporal position",
                "✅ Applies position penalty",
                "✅ Better distribution than basic",
                "✅ Balanced score vs diversity"
            ],
            "cons": [
                "❌ May reduce total score",
                "❌ Penalty weight needs tuning",
                "❌ Not guaranteed uniform distribution",
                "❌ More complex computation"
            ],
            "use_case": "Good balance between score and distribution"
        },
        {
            "name": "SLIDING WINDOW",
            "icon": "🪟", 
            "pros": [
                "✅ Guaranteed temporal spread",
                "✅ Simple and intuitive",
                "✅ Configurable window size",
                "✅ Prevents clustering"
            ],
            "cons": [
                "❌ May miss globally optimal segments",
                "❌ Rigid window structure",
                "❌ Limited flexibility",
                "❌ Window size affects results"
            ],
            "use_case": "When uniform distribution is critical"
        },
        {
            "name": "UNIFORM SAMPLING",
            "icon": "📏",
            "pros": [
                "✅ Perfect temporal distribution", 
                "✅ Predictable spacing",
                "✅ No clustering possible",
                "✅ Simple implementation"
            ],
            "cons": [
                "❌ Ignores content quality",
                "❌ May select poor segments",
                "❌ Fixed intervals",
                "❌ Lowest total scores typically"
            ],
            "use_case": "When temporal uniformity is absolute priority"
        },
        {
            "name": "HYBRID METHOD",
            "icon": "🔄",
            "pros": [
                "✅ Tests all approaches",
                "✅ Selects best result",
                "✅ Adaptive to content",
                "✅ Robust performance"
            ],
            "cons": [
                "❌ Computationally expensive",
                "❌ Complex implementation",
                "❌ Multiple parameters to tune",
                "❌ Longer processing time"
            ],
            "use_case": "Production systems requiring best quality"
        }
    ]
    
    for algo in algorithms:
        print(f"\n{algo['icon']} {algo['name']}")
        print("─" * 50)
        
        print("🟢 ADVANTAGES:")
        for pro in algo['pros']:
            print(f"   {pro}")
            
        print("\n🔴 DISADVANTAGES:")
        for con in algo['cons']:
            print(f"   {con}")
            
        print(f"\n🎯 BEST USE CASE: {algo['use_case']}")

def print_temporal_diversity_metrics():
    """Giải thích các metrics đánh giá temporal diversity"""
    
    print("\n" + "=" * 80)
    print("📊 TEMPORAL DIVERSITY METRICS EXPLAINED")
    print("=" * 80)
    
    metrics = [
        {
            "metric": "MEAN POSITION",
            "formula": "Σ(position_i) / n_segments",
            "range": "[0, 1]",
            "ideal": "0.5 (balanced)",
            "interpretation": {
                "< 0.3": "🔴 Beginning bias - selections concentrated at start",
                "0.3-0.7": "🟡 Balanced - good temporal spread", 
                "> 0.7": "🔴 End bias - selections concentrated at end"
            }
        },
        {
            "metric": "STANDARD DEVIATION",
            "formula": "sqrt(Σ(position_i - mean)² / n)",
            "range": "[0, 0.5]",
            "ideal": "> 0.2 (distributed)",
            "interpretation": {
                "< 0.2": "🔴 Concentrated - poor distribution",
                "0.2-0.3": "🟡 Partial - moderate distribution",
                "> 0.3": "🟢 Distributed - excellent spread"
            }
        },
        {
            "metric": "COVERAGE RANGE", 
            "formula": "max_position - min_position",
            "range": "[0, 1]",
            "ideal": "> 0.6 (good coverage)",
            "interpretation": {
                "< 0.4": "🔴 Poor coverage - limited time span",
                "0.4-0.6": "🟡 Partial coverage - moderate span",
                "> 0.6": "🟢 Good coverage - wide time span"
            }
        },
        {
            "metric": "GAP UNIFORMITY",
            "formula": "std(gaps) / mean(gaps)", 
            "range": "[0, ∞]",
            "ideal": "< 0.5 (uniform gaps)",
            "interpretation": {
                "< 0.3": "🟢 Very uniform - even spacing",
                "0.3-0.5": "🟡 Moderately uniform - acceptable",
                "> 0.5": "🔴 Non-uniform - irregular spacing"
            }
        }
    ]
    
    for metric in metrics:
        print(f"\n📈 {metric['metric']}")
        print("─" * 40)
        print(f"📐 Formula: {metric['formula']}")
        print(f"📊 Range: {metric['range']}")
        print(f"🎯 Ideal: {metric['ideal']}")
        print("📋 Interpretation:")
        for condition, meaning in metric['interpretation'].items():
            print(f"   {condition}: {meaning}")

def print_data_flow():
    """Hiển thị data flow chi tiết"""
    
    print("\n" + "=" * 80)
    print("🌊 DATA FLOW DETAILS")
    print("=" * 80)
    
    data_stages = [
        {
            "stage": "INPUT",
            "data_type": "Video File",
            "shape": "N/A",
            "example": "video.mp4 (1920x1080, 30fps, 120s)",
            "size": "~500MB"
        },
        {
            "stage": "FRAMES",
            "data_type": "Image Tensors", 
            "shape": "(240, 224, 224, 3)",
            "example": "240 frames @ 2fps for 120s video",
            "size": "~150MB"
        },
        {
            "stage": "FEATURES",
            "data_type": "Float Array",
            "shape": "(240, 1024)",
            "example": "GoogleNet Pool5 features",
            "size": "~1MB"
        },
        {
            "stage": "SEGMENTS",
            "data_type": "Integer Pairs",
            "shape": "(n_segments, 2)", 
            "example": "[(0,15), (15,32), (32,58), ...]",
            "size": "~1KB"
        },
        {
            "stage": "SCORES",
            "data_type": "Float Array",
            "shape": "(240,)",
            "example": "[0.1, 0.3, 0.8, 0.2, ...]",
            "size": "~1KB"
        },
        {
            "stage": "SELECTION",
            "data_type": "Binary Array",
            "shape": "(240,)",
            "example": "[0, 0, 1, 1, 0, 1, ...]",
            "size": "~240B"
        }
    ]
    
    print(f"{'STAGE':<12} {'DATA TYPE':<15} {'SHAPE':<20} {'SIZE':<10} {'EXAMPLE'}")
    print("─" * 80)
    
    for stage in data_stages:
        print(f"{stage['stage']:<12} {stage['data_type']:<15} {stage['shape']:<20} {stage['size']:<10} {stage['example']}")

def main():
    """Main function để hiển thị toàn bộ pipeline"""
    
    print_pipeline_overview()
    print_algorithm_comparison() 
    print_temporal_diversity_metrics()
    print_data_flow()
    
    print("\n" + "=" * 80)
    print("🎉 PIPELINE EXPLANATION COMPLETE!")
    print("=" * 80)
    print("\n📁 For visual diagrams, check:")
    print("   • video_summarization_pipeline.png")
    print("   • algorithm_details.png")
    print("\n🚀 Ready to run video summarization!")

if __name__ == "__main__":
    main()
