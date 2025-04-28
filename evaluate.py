"""Evaluate sign language recognition model on validation data."""
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from models import SignLanguageModel
from utils import get_data_loaders, plot_confusion_matrix
from configs import config

def evaluate_model(model, data_loader, criterion, device, class_names):
    """Evaluate model on a dataset."""
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    # Class-wise metrics
    class_correct = {i: 0 for i in range(len(class_names))}
    class_total = {i: 0 for i in range(len(class_names))}
    
    # No gradient calculation needed for evaluation
    with torch.no_grad():
        # Use tqdm for progress bar
        for inputs, labels in tqdm(data_loader, desc='Evaluating'):
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Update statistics
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels).item()
            
            # Collect predictions and labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update class-wise metrics
            for i in range(batch_size):
                label = labels[i].item()
                class_total[label] += 1
                if preds[i] == labels[i]:
                    class_correct[label] += 1
    
    # Calculate overall metrics
    total_samples = len(data_loader.sampler)
    avg_loss = running_loss / total_samples
    accuracy = running_corrects / total_samples
    
    # Calculate per-class metrics
    class_accuracy = {}
    for i in range(len(class_names)):
        if class_total[i] > 0:
            class_accuracy[class_names[i]] = class_correct[i] / class_total[i]
        else:
            class_accuracy[class_names[i]] = 0.0
    
    # Compile metrics
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels)
    }
    
    return metrics

def main():
    """Run evaluation."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders and class names
    _, val_loader, class_names = get_data_loaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        sequence_length=config.SEQUENCE_LENGTH,
        input_size=config.INPUT_SIZE,
        num_workers=2
    )
    
    # Initialize model
    model_path = config.BEST_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return
    
    # Load model
    num_classes = len(class_names)
    model = SignLanguageModel(
        num_classes=num_classes,
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        dropout_rate=config.DROPOUT_RATE,
        bidirectional=config.BIDIRECTIONAL,
        num_lstm_layers=config.NUM_LSTM_LAYERS
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, val_loader, criterion, device, class_names)
    
    # Print overall results
    print("\nEvaluation Results:")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # Print per-class results
    print("\nPer-class accuracy:")
    for cls, acc in metrics['class_accuracy'].items():
        print(f"  {cls}: {acc:.4f}")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['labels'],
        metrics['predictions'],
        class_names,
        save_path='results/confusion_matrix.png'
    )
    
    print(f"\nConfusion matrix saved to results/confusion_matrix.png")

if __name__ == "__main__":
    main()