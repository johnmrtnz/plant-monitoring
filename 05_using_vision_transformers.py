#%%
# ===== IMPORTS =====
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import os


#%%
# ===== CONFIGURATION =====
# Put all config in one place for easy experimentation
CONFIG = {
    'data_dir': "./data/plantvillage dataset/color",
    'model_name': 'google/vit-base-patch16-224',
    'output_dir': './vit_plantvillage',
    'train_batch_size': 16,
    'eval_batch_size': 16,
    'num_epochs': 3,
    'learning_rate': 2e-5,
    'val_split': 0.2,
    'seed': 42,
}

#%%
# ===== DATA LOADING =====
print("📂 Loading dataset...")

# Add error handling
if not os.path.exists(CONFIG['data_dir']):
    raise FileNotFoundError(f"Data directory not found: {CONFIG['data_dir']}")

# Load dataset
dataset = load_dataset("imagefolder", data_dir=CONFIG['data_dir'])

# Split into train/val
split_dataset = dataset['train'].train_test_split(
    test_size=CONFIG['val_split'], 
    seed=CONFIG['seed']
)

trainset = split_dataset['train']
valset = split_dataset['test']

# Get class information
label_names = trainset.features['label'].names
num_classes = len(label_names)

print(f"✅ Dataset loaded successfully")
print(f"   Training samples: {len(trainset):,}")
print(f"   Validation samples: {len(valset):,}")
print(f"   Number of classes: {num_classes}")
print(f"   Sample classes: {label_names[:3]}...")

#%%
# ===== MODEL SETUP =====
print(f"\n🤖 Loading model: {CONFIG['model_name']}")

# Check device availability
device = torch.device(
    'mps' if torch.backends.mps.is_available() else 
    'cuda' if torch.cuda.is_available() else 
    'cpu'
)
print(f"   Device: {device}")

# Load processor
processor = ViTImageProcessor.from_pretrained(CONFIG['model_name'])

# Load model with proper label mapping
model = ViTForImageClassification.from_pretrained(
    CONFIG['model_name'],
    num_labels=num_classes,
    id2label={i: label for i, label in enumerate(label_names)},  # Add this!
    label2id={label: i for i, label in enumerate(label_names)},  # Add this!
    ignore_mismatched_sizes=True
)

print(f"✅ Model loaded with {num_classes} output classes")

#%%
# ===== PREPROCESSING =====
def preprocess(examples):
    """
    Preprocess images for ViT model.
    
    Args:
        examples: Batch of examples from the dataset
        
    Returns:
        Dictionary with 'pixel_values' and 'labels'
    """
    # Process images (don't use return_tensors with with_transform)
    inputs = processor(examples['image'])
    
    # Add labels
    inputs['labels'] = examples['label']
    
    return inputs

# Apply preprocessing
trainset = trainset.with_transform(preprocess)
valset = valset.with_transform(preprocess)

print("✅ Preprocessing applied")

#%%
# ===== EVALUATION METRICS =====
def compute_metrics(eval_pred):
    """
    Compute accuracy, precision, recall, and F1 score.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

#%%
# ===== TRAINING ARGUMENTS =====
training_args = TrainingArguments(
    output_dir=CONFIG['output_dir'],
    
    # Training hyperparameters
    per_device_train_batch_size=CONFIG['train_batch_size'],
    per_device_eval_batch_size=CONFIG['eval_batch_size'],
    num_train_epochs=CONFIG['num_epochs'],
    learning_rate=CONFIG['learning_rate'],
    
    # Evaluation & saving
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',  # Add this!
    greater_is_better=True,            # Add this!
    
    # Logging
    logging_dir=f'{CONFIG["output_dir"]}/logs',
    logging_steps=50,
    logging_strategy='steps',
    
    # Optimization
    save_total_limit=2,  # Only keep 2 best checkpoints
    remove_unused_columns=False,  # Important for image data!
    
    # Reproducibility
    seed=CONFIG['seed'],
    
    # Reporting
    report_to='none',  # Change to 'wandb' if you want experiment tracking
    
    # Performance
    dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues on macOS
    
    # Prevent issues
    push_to_hub=False,
)

print("✅ Training arguments configured")

#%%
# ===== TRAINER =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=trainset,
    eval_dataset=valset,
    compute_metrics=compute_metrics,  # Add this!
)

#%%
# ===== TRAINING =====
print("\n🚀 Starting training...")
print(f"   Epochs: {CONFIG['num_epochs']}")
print(f"   Batch size: {CONFIG['train_batch_size']}")
print(f"   Learning rate: {CONFIG['learning_rate']}")
print("-" * 50)

# Train the model
train_result = trainer.train()

# Print training summary
print("\n" + "=" * 50)
print("📊 Training Complete!")
print("=" * 50)
print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
print(f"Samples per second: {train_result.metrics['train_samples_per_second']:.2f}")

#%%
# ===== EVALUATION =====
print("\n📈 Evaluating on validation set...")

# Evaluate
eval_results = trainer.evaluate()

# Print results
print("\n" + "=" * 50)
print("🎯 Validation Results:")
print("=" * 50)
for key, value in eval_results.items():
    if key.startswith('eval_'):
        metric_name = key.replace('eval_', '').capitalize()
        print(f"{metric_name:.<20} {value:.4f}")

#%%
# ===== SAVE MODEL =====
final_model_path = f'{CONFIG["output_dir"]}/final_model'
trainer.save_model(final_model_path)
processor.save_pretrained(final_model_path)  # Save processor too!

print(f"\n💾 Model saved to: {final_model_path}")

#%%
# ===== PREDICTIONS ON SAMPLE =====
# Test inference on a few samples
print("\n🔍 Testing predictions on sample images...")

model.eval()
sample_predictions = []

for i in range(min(5, len(valset))):
    sample = valset[i]
    
    # Get prediction
    with torch.no_grad():
        inputs = {k: torch.tensor(v).unsqueeze(0) for k, v in sample.items() if k != 'labels'}
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax(-1).item()
    
    true_label = label_names[sample['labels']]
    pred_label = label_names[predicted_class]
    
    print(f"Sample {i+1}: True: {true_label}, Predicted: {pred_label} {'✅' if true_label == pred_label else '❌'}")

print("\n✅ All done!")

#%%
# ===== OPTIONAL: CONFUSION MATRIX =====
# Uncomment to generate confusion matrix

# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Get all predictions
# predictions = trainer.predict(valset)
# y_pred = np.argmax(predictions.predictions, axis=1)
# y_true = predictions.label_ids

# # Create confusion matrix
# cm = confusion_matrix(y_true, y_pred)

# # Plot
# plt.figure(figsize=(12, 10))
# sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.tight_layout()
# plt.savefig(f'{CONFIG["output_dir"]}/confusion_matrix.png')
# print(f"📊 Confusion matrix saved to {CONFIG['output_dir']}/confusion_matrix.png")

# %%