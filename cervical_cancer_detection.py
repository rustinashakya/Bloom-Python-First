"""
Quick Combined Dataset Training
Minimal modification to use combined_dataset.csv

Just replace the data loading section of your existing script with this
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# ============================================================================
# STEP 1: LOAD COMBINED DATASET (Replace your existing data loading)
# ============================================================================

def load_combined_data(csv_path='combined_dataset.csv'):
    """Load combined dataset from CSV"""
    print("\n[Loading Combined Dataset]")
    print("="*80)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"✅ Loaded {len(df)} images")
    print(f"\nDataset breakdown:")
    print(df['dataset_source'].value_counts())
    print(f"\nLabel breakdown:")
    print(df.groupby(['dataset_source', 'label']).size())
    
    # Verify paths exist
    print(f"\nVerifying image paths...")
    valid_rows = []
    for idx, row in df.iterrows():
        if os.path.exists(row['image_path']):
            valid_rows.append(row)
        else:
            if len(valid_rows) == 0:
                print(f"  ⚠️  First invalid path: {row['image_path']}")
    
    df = pd.DataFrame(valid_rows)
    print(f"✅ Valid images: {len(df)}")
    
    return df

# ============================================================================
# STEP 2: SPLIT DATA (Stratified by both dataset and label)
# ============================================================================

def split_combined_data(df, test_size=0.2, val_size=0.125, random_state=42):
    """Split combined dataset ensuring both datasets in each split"""
    print("\n[Splitting Dataset]")
    print("="*80)
    
    # Split by dataset first to ensure representation
    sipakmed_df = df[df['dataset_source'] == 'sipakmed']
    herlev_df = df[df['dataset_source'] == 'herlev']
    
    # Split SIPaKMeD
    sip_train_val, sip_test = train_test_split(
        sipakmed_df, test_size=test_size, 
        stratify=sipakmed_df['label'], random_state=random_state
    )
    sip_train, sip_val = train_test_split(
        sip_train_val, test_size=val_size,
        stratify=sip_train_val['label'], random_state=random_state
    )
    
    # Split Herlev
    her_train_val, her_test = train_test_split(
        herlev_df, test_size=test_size,
        stratify=herlev_df['label'], random_state=random_state
    )
    her_train, her_val = train_test_split(
        her_train_val, test_size=val_size,
        stratify=her_train_val['label'], random_state=random_state
    )
    
    # Combine splits
    train_df = pd.concat([sip_train, her_train], ignore_index=True)
    val_df = pd.concat([sip_val, her_val], ignore_index=True)
    test_df = pd.concat([sip_test, her_test], ignore_index=True)
    
    print(f"Train: {len(train_df)} (SIPaKMeD: {len(sip_train)}, Herlev: {len(her_train)})")
    print(f"Val:   {len(val_df)} (SIPaKMeD: {len(sip_val)}, Herlev: {len(her_val)})")
    print(f"Test:  {len(test_df)} (SIPaKMeD: {len(sip_test)}, Herlev: {len(her_test)})")
    
    return train_df, val_df, test_df

# ============================================================================
# STEP 3: CREATE DATA GENERATORS
# ============================================================================

def create_generators(train_df, val_df, test_df, img_size=224, batch_size=16):
    """Create data generators"""
    print("\n[Creating Data Generators]")
    print("="*80)
    
    # Convert labels to strings for Keras
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df['label_str'] = train_df['label'].astype(str)
    val_df['label_str'] = val_df['label'].astype(str)
    test_df['label_str'] = test_df['label'].astype(str)
    
    # Strong augmentation for combined dataset
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.3,
        shear_range=0.2,
        brightness_range=[0.7, 1.3],  # Important for different datasets
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col='image_path', y_col='label_str',
        target_size=(img_size, img_size),
        batch_size=batch_size, class_mode='binary', shuffle=True
    )
    
    val_gen = val_test_datagen.flow_from_dataframe(
        val_df, x_col='image_path', y_col='label_str',
        target_size=(img_size, img_size),
        batch_size=batch_size, class_mode='binary', shuffle=False
    )
    
    test_gen = val_test_datagen.flow_from_dataframe(
        test_df, x_col='image_path', y_col='label_str',
        target_size=(img_size, img_size),
        batch_size=batch_size, class_mode='binary', shuffle=False
    )
    
    print(f"✅ Train: {len(train_gen)} batches")
    print(f"✅ Val:   {len(val_gen)} batches")
    print(f"✅ Test:  {len(test_gen)} batches")
    
    return train_gen, val_gen, test_gen, train_df, val_df, test_df

# ============================================================================
# STEP 4: COMPUTE CLASS WEIGHTS
# ============================================================================

def get_class_weights(train_df):
    """Compute class weights for imbalanced data"""
    print("\n[Computing Class Weights]")
    print("="*80)
    
    labels = train_df['label'].values
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(labels), 
        y=labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"Class 0 (Normal):   weight = {class_weight_dict[0]:.3f}")
    print(f"Class 1 (Abnormal): weight = {class_weight_dict[1]:.3f}")
    
    return class_weight_dict

# ============================================================================
# STEP 5: EVALUATE ON EACH DATASET SEPARATELY
# ============================================================================

def evaluate_by_dataset(model, test_df, img_size=224, batch_size=16):
    """Evaluate separately on SIPaKMeD and Herlev"""
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
    print("\n" + "="*80)
    print("EVALUATION BY DATASET")
    print("="*80)
    
    results = {}
    
    for dataset_name in ['sipakmed', 'herlev']:
        dataset_df = test_df[test_df['dataset_source'] == dataset_name].copy()
        
        if len(dataset_df) == 0:
            continue
        
        print(f"\n{'='*80}")
        print(f"{dataset_name.upper()} TEST SET")
        print(f"{'='*80}")
        
        # Create generator
        dataset_df['label_str'] = dataset_df['label'].astype(str)
        datagen = ImageDataGenerator(rescale=1./255)
        gen = datagen.flow_from_dataframe(
            dataset_df, x_col='image_path', y_col='label_str',
            target_size=(img_size, img_size),
            batch_size=batch_size, class_mode='binary', shuffle=False
        )
        
        # Predict
        y_pred_proba = model.predict(gen, verbose=0).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        y_true = dataset_df['label'].values
        
        # Metrics
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Normal', 'Abnormal']))
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / len(y_true)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nKey Metrics:")
        print(f"  Accuracy:    {accuracy*100:.2f}%")
        print(f"  Sensitivity: {sensitivity*100:.2f}%")
        print(f"  Specificity: {specificity*100:.2f}%")
        print(f"  Missed Cancers: {fn}")
        
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            print(f"  ROC-AUC: {roc_auc:.4f}")
        except:
            roc_auc = 0
        
        results[dataset_name] = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'missed_cancers': int(fn)
        }
    
    return results

# ============================================================================
# USAGE EXAMPLE - Replace your main() function with this
# ============================================================================

def main():
    """Main training pipeline using combined dataset"""
    
    # Configuration
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS_FROZEN = 50
    EPOCHS_FINETUNE = 30
    
    # 1. Load combined dataset
    df = load_combined_data('combined_dataset.csv')
    
    # 2. Split data
    train_df, val_df, test_df = split_combined_data(df)
    
    # 3. Create generators
    train_gen, val_gen, test_gen, train_df, val_df, test_df = create_generators(
        train_df, val_df, test_df, IMG_SIZE, BATCH_SIZE
    )
    
    # 4. Compute class weights
    class_weights = get_class_weights(train_df)
    
    # 5. Build model (use your existing model building code)
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    
    base_model = VGG16(weights='imagenet', include_top=False, 
                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    print("\n[Training Phase 1: Frozen Base]")
    print("="*80)
    
    # 6. Train with class weights
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    
    callbacks = [
        ModelCheckpoint('best_model_combined.h5', save_best_only=True, monitor='val_loss'),
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-7)
    ]
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS_FROZEN,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,  # ← Important!
        verbose=1
    )
    
    # 7. Fine-tuning
    print("\n[Training Phase 2: Fine-tuning]")
    print("="*80)
    
    base_model.trainable = True
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='binary_crossentropy',
        metrics=['accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    callbacks = [
        ModelCheckpoint('final_model_combined.h5', save_best_only=True, monitor='val_loss'),
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-7)
    ]
    
    history_fine = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # 8. Evaluate on both datasets
    results = evaluate_by_dataset(model, test_df, IMG_SIZE, BATCH_SIZE)
    
    # 9. Save results
    import json
    with open('combined_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\n📊 Results Summary:")
    for dataset, metrics in results.items():
        print(f"\n{dataset.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Sensitivity: {metrics['sensitivity']*100:.2f}%")
        print(f"  Specificity: {metrics['specificity']*100:.2f}%")
        print(f"  Missed Cancers: {metrics['missed_cancers']}")
    
    print(f"\n💾 Model saved to: final_model_combined.h5")
    print(f"🎯 This model now works on BOTH SIPaKMeD and Herlev!")
    
    return model, results

# ============================================================================
# RUN IT!
# ============================================================================

if __name__ == "__main__":
    model, results = main()