import os
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure static folder for uploads
app.static_folder = 'uploads'
app.static_url_path = '/uploads'

# Load your trained YOLOv8 model
model = YOLO('best .pt')  # Replace with your model path

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
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(image_path))
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    return output_path

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
    
    # Release resources
    cap.release()
    out.release()
    
    return output_path

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
                output_path = process_video(file_path)
            else:
                output_path = process_image(file_path)
            
            # Get the relative URL for the processed file
            output_filename = os.path.basename(output_path)
            output_url = url_for('static', filename=output_filename)
            
            return jsonify({
                'success': True,
                'output_path': output_url,
                'message': 'File processed successfully'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True) 