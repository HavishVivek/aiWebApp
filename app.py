from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model("./arduino_due_detector_transfer.h5")

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Load and preprocess the image for prediction."""
    img = image.load_img(img_path, target_size=(150, 150))  # Resize to 150x150
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

@app.route('/')
def index():
    """Render the home page with the upload form."""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Handle image upload and make predictions."""
    if 'file' not in request.files:
        return "No file part in the request", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected for upload", 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Predict using the model
        prediction = model.predict(img_array)
        confidence = prediction[0][0]  # Confidence score

        # Classification based on threshold
        if confidence < 0.5:
            result = f"Arduino Due detected with {confidence * 100:.2f}% confidence."
        else:
            result = f"No Arduino Due detected with {100 - confidence * 100:.2f}% confidence."
        
        # Clean up the uploaded file after processing
        os.remove(file_path)
        
        return result
    else:
        return "Invalid file type. Please upload an image file.", 400

if __name__ == '__main__':
    # Ensure the uploads directory exists
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)