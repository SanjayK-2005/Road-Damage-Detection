import os
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import json
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure static folder for uploads
app.static_folder = 'uploads'
app.static_url_path = '/uploads'

# Load your trained YOLOv8 model
model = YOLO('best.pt')  # Replace with your model path

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    # Run inference on the image
    results = model(image_path)
    
    # Get the first result
    result = results[0]
    
    # Convert the image to numpy array
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw boxes and labels
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f'{result.names[cls]} {conf:.2f}'
        
        # Draw rectangle (BGR format: (0, 0, 255) for red)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        
        # Draw label (BGR format: (255, 255, 255) for white)
        cv2.putText(img, label, (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create output video writer
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference on the frame
        results = model(frame)
        result = results[0]
        
        # Draw boxes and labels
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f'{result.names[cls]} {conf:.2f}'
            
            # Draw rectangle (BGR format: (0, 0, 255) for red)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            
            # Draw label (BGR format: (255, 255, 255) for white)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Write the frame to output video
        out.write(frame)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frames.append(base64.b64encode(buffer).decode('utf-8'))
    
    # Release resources
    cap.release()
    out.release()
    
    return frames

def get_model_metrics():
    """Get model performance metrics"""
    try:
        # Run validation
        results = model.val(
            data='data.yaml',
            conf=0.25,
            iou=0.45,
            verbose=False,
            save_json=True,
            save_hybrid=True,
            plots=True
        )
        
        metrics = {
            'precision': float(results.box.precision),
            'recall': float(results.box.recall),
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'f1-score': float(results.box.f1),
            'class_names': results.names,
            'per_class_precision': results.box.precision.tolist(),
            'per_class_recall': results.box.recall.tolist(),
            'per_class_map50': results.box.map50.tolist(),
            'per_class_map': results.box.map.tolist()
        }
        
        # Save metrics to JSON file
        with open('model_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
            
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return None

@app.route('/metrics')
def show_metrics():
    metrics = get_model_metrics()
    if metrics:
        return render_template('metrics.html', metrics=metrics)
    return jsonify({'error': 'Failed to calculate metrics'}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Process the file based on its type
            if filename.lower().endswith(('.mp4', '.avi')):
                frames = process_video(file_path)
                return jsonify({
                    'success': True,
                    'frames': frames,
                    'message': 'Video processed successfully'
                })
            else:
                base64_image = process_image(file_path)
                return jsonify({
                    'success': True,
                    'image': base64_image,
                    'message': 'Image processed successfully'
                })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)