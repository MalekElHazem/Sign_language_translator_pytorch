"""Diagnostic sign language detection with debugging info."""
import cv2
import torch
import numpy as np
from torchvision import transforms
from collections import deque
import os
import time
import matplotlib.pyplot as plt
from PIL import Image

from models import SignLanguageModel
from configs import config # Import config directly

def load_class_names(file_path):
    """Load class names from file."""
    if not os.path.exists(file_path):
        print(f"Error: Class names file not found at {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error reading class names file {file_path}: {e}")
        return None

def load_model(model_path, num_classes, device):
    """Load a trained model."""
    model = SignLanguageModel(
        num_classes=num_classes,
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        dropout_rate=config.DROPOUT_RATE,
        bidirectional=config.BIDIRECTIONAL,
        num_lstm_layers=config.NUM_LSTM_LAYERS
    ).to(device)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    try:
        # Use weights_only=True for added security if PyTorch version supports it
        if hasattr(torch, 'load') and 'weights_only' in torch.load.__code__.co_varnames:
             model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        else:
             model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model state dict from {model_path}: {e}")
        return None

def save_debug_frame(frame_orig_bgr, frame_processed_tensor=None, sequence_number=0, save_dir="debug_frames"):
    """Save original and processed frames for debugging, correcting color."""
    os.makedirs(save_dir, exist_ok=True)

    # Save original ROI frame (BGR)
    cv2.imwrite(os.path.join(save_dir, f"orig_roi_{sequence_number:04d}.jpg"), frame_orig_bgr)

    # Save processed frame if provided (should be RGB tensor before normalization)
    if frame_processed_tensor is not None:
        # Check tensor shape (should be [3, H, W] for RGB)
        if frame_processed_tensor.shape[0] == 3:
            # Convert RGB tensor to numpy and denormalize
            img = frame_processed_tensor.cpu().numpy().transpose(1, 2, 0) # H, W, C
            # Denormalize using standard ImageNet mean/std
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img * 255, 0, 255).astype(np.uint8)

            # --- FIX: Convert RGB numpy array to BGR for OpenCV imwrite ---
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # --- End Fix ---

            cv2.imwrite(os.path.join(save_dir, f"proc_rgb_{sequence_number:04d}.jpg"), img_bgr) # Save BGR
        elif frame_processed_tensor.shape[0] == 1:
             # Handle grayscale case if you ever switch back
             img = frame_processed_tensor.cpu().numpy().squeeze()
             img = img * 0.5 + 0.5 # Assuming mean=0.5, std=0.5
             img = np.clip(img * 255, 0, 255).astype(np.uint8)
             cv2.imwrite(os.path.join(save_dir, f"proc_gray_{sequence_number:04d}.jpg"), img)
        else:
             print(f"Warning: Unexpected processed frame shape {frame_processed_tensor.shape} in save_debug_frame")


def diagnostic_detection():
    """Run diagnostic sign language detection with lower thresholds and debugging."""
    model_path = config.BEST_MODEL_PATH
    class_names_file = config.CLASS_NAMES_FILE

    # Load class names
    class_names = load_class_names(class_names_file)
    if class_names is None: return

    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes: {', '.join(class_names)}")

    # Find neutral class index
    neutral_idx = -1
    if 'neutral' in class_names:
        neutral_idx = class_names.index('neutral')
        print(f"'neutral' class found at index: {neutral_idx}")
    else:
        print("Warning: 'neutral' class not found. Handicap will not be applied.")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(model_path, num_classes, device)
    if model is None: return
    print(f"Model loaded from {model_path}")

    # Set up video capture
    cap = None
    for camera_index in [0, 1, 2]:
        try:
            print(f"Trying camera index {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Successfully opened camera {camera_index}")
                break
            else:
                cap.release()
        except Exception as e:
            print(f"Error with camera {camera_index}: {e}")

    if cap is None or not cap.isOpened():
        print("Failed to open any camera.")
        return

    # Define transformations - MUST MATCH validation transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        # NO Augmentations like Flip/ColorJitter here
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Buffer for holding the sequence of frames
    frame_buffer = deque(maxlen=config.SEQUENCE_LENGTH)

    # Prediction history for temporal smoothing
    prediction_history = deque(maxlen=config.HISTORY_SIZE)

    # Motion detection variables
    prev_frame = None
    motion_history = deque(maxlen=5)

    # MUCH LOWER motion threshold for diagnostics
    motion_threshold = config.MOTION_THRESHOLD / 10.0
    # Lower confidence threshold
    confidence_threshold = 0.2 # Lower for diagnostics

    # Debug variables
    debug_frames_saved = 0
    last_prediction_time = time.time()
    frame_count = 0

    print("Starting diagnostic detection. Press 'q' to quit.")
    print(f"Motion threshold: {motion_threshold} (lowered)")
    print(f"Confidence threshold: {confidence_threshold} (lowered)")
    print("Press '+' to increase motion threshold, '-' to decrease it")
    print("Debug frames will be saved to 'debug_frames' directory")

    # Statistics tracking
    predictions = {class_name: 0 for class_name in class_names}

    try:
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break

            frame_count += 1
            # Get frame dimensions
            frame_height, frame_width, _ = frame.shape

            # Define ROI (region of interest)
            roi_x = int(frame_width * 0.05)
            roi_y = int(frame_height * 0.05)
            roi_w = int(frame_width * 0.9)
            roi_h = int(frame_height * 0.9)

            # Create a copy for display
            display_frame = frame.copy()

            # Draw ROI rectangle
            cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

            # Extract ROI for motion detection and processing
            roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] # This is BGR

            # Calculate motion between frames
            motion_score = 0
            if prev_frame is not None:
                prev_roi = prev_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                curr_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
                frame_diff = cv2.absdiff(curr_gray, prev_gray)
                motion_score = np.sum(frame_diff) / (roi_w * roi_h * 255)
                motion_history.append(motion_score)
                diff_color = cv2.applyColorMap((frame_diff * 5).astype(np.uint8), cv2.COLORMAP_JET)
                small_diff = cv2.resize(diff_color, (160, 120))
                display_frame[10:130, 10:170] = small_diff

            prev_frame = frame.copy()

            # Process ROI for sign detection
            roi_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB) # Convert BGR ROI to RGB
            transformed_frame = transform(roi_rgb) # Apply the corrected transform

            # Save debug frame occasionally
            if frame_count % 30 == 0: # Every 30 frames (~1 second)
                # Save original ROI (BGR) and processed tensor (RGB normalized)
                save_debug_frame(roi_frame, transformed_frame, debug_frames_saved)
                debug_frames_saved += 1

            # Add to frame buffer
            frame_buffer.append(transformed_frame)

            # Make a prediction once we have enough frames
            text = f"Collecting frames... ({len(frame_buffer)}/{config.SEQUENCE_LENGTH})"
            confidence_score = 0
            predicted_class = None
            probabilities = None

            # Calculate average motion
            avg_motion = sum(motion_history) / len(motion_history) if motion_history else 0

            # --- Prediction Logic (Simplified for Diagnostics) ---
            if len(frame_buffer) == config.SEQUENCE_LENGTH:
                # Convert buffer to tensor
                input_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)

                # Get model prediction
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)

                    # --- Apply Neutral Handicap ---
                    if neutral_idx != -1 and config.NEUTRAL_HANDICAP > 0:
                        probs[0, neutral_idx] = max(0.0, probs[0, neutral_idx] - config.NEUTRAL_HANDICAP)
                        probs = probs / probs.sum(dim=1, keepdim=True) # Re-normalize
                    # --- End Handicap ---

                    top_prob, top_class = torch.max(probs, 1)
                    predicted_idx = top_class.item()
                    confidence = top_prob.item()
                    probabilities = probs[0].cpu().numpy()

                confidence_score = confidence * 100

                # Apply confidence threshold (lower for diagnostic)
                if confidence > confidence_threshold:
                    predicted_class = class_names[predicted_idx]
                    predictions[predicted_class] += 1 # Record prediction
                    prediction_history.append((predicted_idx, confidence_score))

                    # Temporal Smoothing (simplified for diagnostics)
                    if len(prediction_history) >= 1: # Show if detected once
                        text = f"Predicted: {predicted_class} ({confidence_score:.1f}%)"
                    else:
                        text = "Uncertain prediction"
                else:
                    text = f"Low confidence: {confidence_score:.1f}%"
                    prediction_history.clear() # Clear history if low confidence
            # --- End Prediction Logic ---

            # Display info
            cv2.putText(display_frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display motion score
            motion_text = f"Motion: {avg_motion:.6f} (Thresh: {motion_threshold:.6f})"
            cv2.putText(display_frame, motion_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Display buffer status
            buffer_status = f"Buffer: {len(frame_buffer)}/{config.SEQUENCE_LENGTH}"
            cv2.putText(display_frame, buffer_status, (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Display prediction statistics
            stats_text = f"Stats: {' | '.join([f'{k}:{v}' for k,v in predictions.items() if v > 0])}"
            cv2.putText(display_frame, stats_text, (10, frame_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Display all class probabilities if available
            if probabilities is not None:
                bar_height = 15; bar_width = 100; bar_gap = 5
                start_y = frame_height - (5 * (bar_height + bar_gap)) - 30 # Position from bottom
                sorted_indices = np.argsort(probabilities)[::-1]
                for i, idx in enumerate(sorted_indices[:5]):
                    label_text = f"{class_names[idx]}: {probabilities[idx]*100:.1f}%"
                    text_y = start_y + i*(bar_height+bar_gap)
                    cv2.putText(display_frame, label_text, (10, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    bar_length = int(probabilities[idx] * bar_width)
                    cv2.rectangle(display_frame, (150, text_y - bar_height + 3),
                                 (150 + bar_length, text_y + 3),
                                 (0, 255, 0) if idx == predicted_idx and confidence > confidence_threshold else (0, 165, 255), -1)

            # Display instructions
            cv2.putText(display_frame, "Make signs inside the green box", (roi_x, roi_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display frame
            cv2.imshow('Diagnostic Sign Language Detection', display_frame)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('+'):
                motion_threshold *= 1.5
                print(f"Increased motion threshold to: {motion_threshold:.6f}")
            elif key == ord('-'):
                motion_threshold /= 1.5
                print(f"Decreased motion threshold to: {motion_threshold:.6f}")
            elif key == ord('s'): # Manual save
                save_debug_frame(roi_frame, transformed_frame, debug_frames_saved, "manual_debug")
                debug_frames_saved += 1
                print(f"Saved manual debug frame {debug_frames_saved}")

    except KeyboardInterrupt: print("\nDetection interrupted by user.")
    except Exception as e:
        print(f"Error in detection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if cap is not None: cap.release()
        cv2.destroyAllWindows()
        print("\nDetection statistics:")
        for class_name, count in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count} predictions")
        print(f"\nDebug frames saved: {debug_frames_saved}")

def main():
    """Run diagnostic detection."""
    diagnostic_detection()

if __name__ == "__main__":
    main()