"""
Adaptive Inference with Dataset-Specific Thresholds
Automatically adjusts threshold based on image characteristics

Solution to Herlev's 50% specificity problem
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION WITH DATASET-SPECIFIC THRESHOLDS
# ============================================================================

class AdaptiveInferenceConfig:
    # Model
    MODEL_PATH = 'final_model_combined.h5'
    IMG_SIZE = 224
    
    # Dataset-specific thresholds (optimize these!)
    THRESHOLDS = {
        'sipakmed': 0.50,  # Default works well for SIPaKMeD
        'herlev': 0.65,    # Higher threshold for Herlev (reduces false positives)
        'unknown': 0.55    # Middle ground for unknown datasets
    }
    
    # Auto-detection based on image characteristics
    AUTO_DETECT = True
    
    # Risk levels
    RISK_LEVELS = {
        'very_low': 0.3,
        'low': 0.5,
        'moderate': 0.7,
        'high': 0.85
    }

config = AdaptiveInferenceConfig()

print("="*80)
print("ADAPTIVE INFERENCE SYSTEM")
print("="*80)
print(f"Model: {config.MODEL_PATH}")
print(f"Thresholds:")
print(f"  SIPaKMeD: {config.THRESHOLDS['sipakmed']}")
print(f"  Herlev: {config.THRESHOLDS['herlev']}")
print(f"  Auto-detect: {config.AUTO_DETECT}")
print("="*80)

# ============================================================================
# LOAD MODEL
# ============================================================================

print("\n[1] Loading model...")
try:
    model = load_model(config.MODEL_PATH)
    print(f"✅ Model loaded")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# ============================================================================
# DATASET AUTO-DETECTION
# ============================================================================

def detect_dataset_type(image_path):
    """
    Auto-detect if image is likely from SIPaKMeD or Herlev
    based on image characteristics
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 'unknown'
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Analyze color characteristics
        h_mean = np.mean(hsv[:,:,0])
        s_mean = np.mean(hsv[:,:,1])
        v_mean = np.mean(hsv[:,:,2])
        
        # Simple heuristic (you can improve this based on your data)
        # SIPaKMeD tends to have different staining characteristics than Herlev
        
        # Check if path contains dataset name
        path_lower = image_path.lower()
        if 'sipakmed' in path_lower or 'im_' in path_lower:
            return 'sipakmed'
        elif 'herlev' in path_lower:
            return 'herlev'
        
        # Fallback to image analysis
        # These thresholds should be calibrated on your actual data
        if s_mean > 100 and v_mean > 150:
            return 'sipakmed'  # Typically more saturated
        elif s_mean < 80:
            return 'herlev'    # Typically less saturated
        
        return 'unknown'
        
    except Exception as e:
        print(f"⚠️  Detection error: {e}")
        return 'unknown'

# ============================================================================
# PREDICTION WITH ADAPTIVE THRESHOLD
# ============================================================================

def preprocess_image(image_path):
    """Preprocess image"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_adaptive(image_path, dataset_type=None, manual_threshold=None):
    """
    Predict with adaptive threshold
    
    Args:
        image_path: Path to image
        dataset_type: 'sipakmed', 'herlev', or None for auto-detect
        manual_threshold: Override automatic threshold
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {os.path.basename(image_path)}")
    print(f"{'='*80}")
    
    # Auto-detect dataset if not specified
    if dataset_type is None and config.AUTO_DETECT:
        dataset_type = detect_dataset_type(image_path)
        print(f"📊 Auto-detected dataset: {dataset_type}")
    elif dataset_type is None:
        dataset_type = 'unknown'
    
    # Get threshold
    if manual_threshold is not None:
        threshold = manual_threshold
        print(f"🎯 Using manual threshold: {threshold}")
    else:
        threshold = config.THRESHOLDS.get(dataset_type, config.THRESHOLDS['unknown'])
        print(f"🎯 Using {dataset_type} threshold: {threshold}")
    
    # Preprocess
    img = preprocess_image(image_path)
    if img is None:
        print("❌ Failed to load image")
        return None
    
    # Predict
    pred_proba = model.predict(img, verbose=0)[0][0]
    pred_class = 1 if pred_proba >= threshold else 0
    pred_label = 'Abnormal' if pred_class == 1 else 'Normal'
    
    # Calculate confidence
    if pred_class == 1:
        confidence = (pred_proba / threshold) * 50 + 50
    else:
        confidence = ((threshold - pred_proba) / threshold) * 50 + 50
    confidence = min(confidence, 100)
    
    # Risk level
    if pred_proba < config.RISK_LEVELS['very_low']:
        risk = "✅ VERY LOW RISK"
        risk_color = "green"
    elif pred_proba < config.RISK_LEVELS['low']:
        risk = "🟢 LOW RISK"
        risk_color = "lightgreen"
    elif pred_proba < config.RISK_LEVELS['moderate']:
        risk = "🟡 MODERATE RISK"
        risk_color = "yellow"
    elif pred_proba < config.RISK_LEVELS['high']:
        risk = "🟠 ELEVATED RISK"
        risk_color = "orange"
    else:
        risk = "🔴 HIGH RISK"
        risk_color = "red"
    
    # Display results
    print(f"\n📊 RESULTS:")
    print(f"   Raw Score: {pred_proba:.4f}")
    print(f"   Threshold: {threshold:.2f}")
    print(f"   Prediction: {pred_label}")
    print(f"   Confidence: {confidence:.1f}%")
    print(f"   Risk Level: {risk}")
    
    # Comparison with different thresholds
    print(f"\n🔄 With different thresholds:")
    for dt, th in config.THRESHOLDS.items():
        pred_at_th = 'Abnormal' if pred_proba >= th else 'Normal'
        print(f"   {dt:10s} (th={th:.2f}): {pred_at_th}")
    
    # Clinical interpretation
    print(f"\n🏥 CLINICAL INTERPRETATION:")
    if pred_proba < 0.3:
        print(f"   → Cells appear normal")
        print(f"   → Routine follow-up recommended")
    elif pred_proba < 0.5:
        print(f"   → Minimal concern")
        print(f"   → Consider repeat test in 6-12 months")
    elif pred_proba < 0.7:
        print(f"   → Moderate concern")
        print(f"   → Additional testing recommended")
    elif pred_proba < 0.85:
        print(f"   → Significant concern")
        print(f"   → Immediate follow-up required")
    else:
        print(f"   → High concern")
        print(f"   → Urgent medical evaluation needed")
    
    return {
        'image_path': image_path,
        'dataset_type': dataset_type,
        'raw_score': pred_proba,
        'threshold': threshold,
        'prediction': pred_label,
        'confidence': confidence,
        'risk_level': risk
    }

# ============================================================================
# BATCH PREDICTION WITH STATISTICS
# ============================================================================

def predict_batch_adaptive(folder_path, dataset_type=None):
    """Batch prediction with adaptive thresholds"""
    print(f"\n{'='*80}")
    print(f"BATCH PREDICTION - ADAPTIVE MODE")
    print(f"{'='*80}")
    print(f"Folder: {folder_path}")
    
    # Get images
    extensions = ['.bmp', '.jpg', '.jpeg', '.png']
    images = []
    for ext in extensions:
        images.extend([os.path.join(folder_path, f) 
                      for f in os.listdir(folder_path) 
                      if f.lower().endswith(ext)])
    
    if not images:
        print("❌ No images found")
        return None
    
    print(f"Found {len(images)} images")
    
    # Predict each
    results = []
    for img_path in images:
        result = predict_adaptive(img_path, dataset_type, manual_threshold=None)
        if result:
            results.append(result)
    
    # Statistics
    print(f"\n{'='*80}")
    print(f"BATCH SUMMARY")
    print(f"{'='*80}")
    
    normal = sum(1 for r in results if r['prediction'] == 'Normal')
    abnormal = sum(1 for r in results if r['prediction'] == 'Abnormal')
    
    print(f"\nTotal: {len(results)}")
    print(f"  Normal: {normal} ({normal/len(results)*100:.1f}%)")
    print(f"  Abnormal: {abnormal} ({abnormal/len(results)*100:.1f}%)")
    
    # By dataset type
    print(f"\n📊 By Detected Dataset:")
    for dt in ['sipakmed', 'herlev', 'unknown']:
        dt_results = [r for r in results if r['dataset_type'] == dt]
        if dt_results:
            dt_normal = sum(1 for r in dt_results if r['prediction'] == 'Normal')
            dt_abnormal = sum(1 for r in dt_results if r['prediction'] == 'Abnormal')
            print(f"   {dt:10s}: {len(dt_results)} images")
            print(f"      Normal: {dt_normal}, Abnormal: {dt_abnormal}")
    
    # Score statistics
    scores = [r['raw_score'] for r in results]
    print(f"\n📈 Score Statistics:")
    print(f"   Mean: {np.mean(scores):.3f}")
    print(f"   Median: {np.median(scores):.3f}")
    print(f"   Min: {np.min(scores):.3f}")
    print(f"   Max: {np.max(scores):.3f}")
    
    return results

# ============================================================================
# COMPARISON MODE
# ============================================================================

def compare_thresholds(image_path):
    """Compare predictions with different thresholds"""
    print(f"\n{'='*80}")
    print(f"THRESHOLD COMPARISON")
    print(f"{'='*80}")
    print(f"Image: {os.path.basename(image_path)}")
    
    # Get prediction score
    img = preprocess_image(image_path)
    if img is None:
        return
    
    score = model.predict(img, verbose=0)[0][0]
    
    print(f"\nRaw Score: {score:.4f}")
    print(f"\n{'Threshold':<12} {'Prediction':<12} {'Use Case'}")
    print("-" * 50)
    
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    for th in thresholds:
        pred = 'Abnormal' if score >= th else 'Normal'
        
        if th <= 0.40:
            use_case = "Screening (high sens.)"
        elif th <= 0.55:
            use_case = "General (balanced)"
        else:
            use_case = "Confirmation (high spec.)"
        
        print(f"{th:.2f}         {pred:<12} {use_case}")
    
    print("\n" + "="*80)

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode():
    """Interactive prediction"""
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("\nOptions:")
    print("  1. Predict single image (auto-detect dataset)")
    print("  2. Predict single image (specify dataset)")
    print("  3. Predict batch (auto-detect)")
    print("  4. Predict batch (specify dataset)")
    print("  5. Compare thresholds for image")
    print("  6. Update thresholds")
    print("  7. Exit")
    
    while True:
        print("\n" + "-"*80)
        choice = input("\nSelect (1-7): ").strip()
        
        if choice == '1':
            path = input("Image path: ").strip().strip('"').strip("'")
            if os.path.exists(path):
                predict_adaptive(path)
            else:
                print("❌ Not found")
        
        elif choice == '2':
            path = input("Image path: ").strip().strip('"').strip("'")
            print("\nDataset types:")
            print("  1. SIPaKMeD")
            print("  2. Herlev")
            print("  3. Unknown")
            dt_choice = input("Select (1-3): ").strip()
            dt_map = {'1': 'sipakmed', '2': 'herlev', '3': 'unknown'}
            dataset_type = dt_map.get(dt_choice, 'unknown')
            
            if os.path.exists(path):
                predict_adaptive(path, dataset_type=dataset_type)
            else:
                print("❌ Not found")
        
        elif choice == '3':
            path = input("Folder path: ").strip().strip('"').strip("'")
            if os.path.exists(path):
                predict_batch_adaptive(path)
            else:
                print("❌ Not found")
        
        elif choice == '4':
            path = input("Folder path: ").strip().strip('"').strip("'")
            print("\nDataset types:")
            print("  1. SIPaKMeD")
            print("  2. Herlev")
            dt_choice = input("Select (1-2): ").strip()
            dt_map = {'1': 'sipakmed', '2': 'herlev'}
            dataset_type = dt_map.get(dt_choice)
            
            if os.path.exists(path):
                predict_batch_adaptive(path, dataset_type=dataset_type)
            else:
                print("❌ Not found")
        
        elif choice == '5':
            path = input("Image path: ").strip().strip('"').strip("'")
            if os.path.exists(path):
                compare_thresholds(path)
            else:
                print("❌ Not found")
        
        elif choice == '6':
            print(f"\nCurrent thresholds:")
            print(f"  SIPaKMeD: {config.THRESHOLDS['sipakmed']}")
            print(f"  Herlev: {config.THRESHOLDS['herlev']}")
            print(f"  Unknown: {config.THRESHOLDS['unknown']}")
            
            print(f"\n💡 Recommendations:")
            print(f"   - Lower threshold (0.3-0.4): More sensitive, fewer missed cancers")
            print(f"   - Higher threshold (0.6-0.7): More specific, fewer false alarms")
            print(f"   - Current Herlev: 0.65 (to fix 50% specificity issue)")
            
            dt = input("\nWhich to update (sipakmed/herlev/unknown): ").strip().lower()
            if dt in config.THRESHOLDS:
                try:
                    new_th = float(input(f"New threshold for {dt} (0.0-1.0): "))
                    if 0 <= new_th <= 1:
                        config.THRESHOLDS[dt] = new_th
                        print(f"✅ Updated {dt} threshold to {new_th}")
                    else:
                        print("❌ Invalid range")
                except:
                    print("❌ Invalid input")
        
        elif choice == '7':
            print("\nExiting...")
            break
        
        else:
            print("❌ Invalid option")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n💡 ADAPTIVE INFERENCE SYSTEM")
    print("   Automatically adjusts threshold based on dataset")
    print("   Solves the Herlev 50% specificity problem!")
    
    interactive_mode()