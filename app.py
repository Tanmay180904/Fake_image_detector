"""
Fake Image Detection System - Flask Backend
Main application file for handling image uploads and predictions
"""
import os
import logging

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, send_from_directory, session
)
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import models
import matplotlib.cm as cm

# -------------------------------------------------
# Configure logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Initialize Flask app
# -------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

MODEL_PATH = os.path.join('model', 'fake_detector_rgb.h5')
IMG_SIZE = 224

# Load model
model = load_model(MODEL_PATH)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('model', exist_ok=True)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    )

def preprocess_image(image_path):
    """
    Preprocess for ResNet50:
    - Convert to RGB
    - Resize to IMG_SIZE x IMG_SIZE
    - Apply preprocess_input
    - Add batch dimension
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img).astype(np.float32)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
        logger.info(f"Image preprocessed successfully: Shape {img_array.shape}")
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict_image(image_array):
    """
    Use trained CNN model to predict real/fake.
    image_array: (1, IMG_SIZE, IMG_SIZE, 3)
    """
    try:
        prob = float(model.predict(image_array)[0][0])  # prob = P(real)

        if prob >= 0.5:
            prediction = 'Real'
            confidence = int(prob * 100)
        else:
            prediction = 'Fake'
            confidence = int((1.0 - prob) * 100)

        # Clamp confidence to [50, 100]
        confidence = max(50, min(100, confidence))
        logger.info(f"Prediction: {prediction}, Confidence: {confidence}%")
        return prediction, confidence
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def perform_ela(input_path, output_path, quality=90):
    """
    Create an Error Level Analysis (ELA) image.

    input_path: original uploaded image path
    output_path: where to save ELA image (PNG)
    """
    try:
        original = Image.open(input_path).convert('RGB')

        # Step 1: re-save as JPEG to introduce compression
        temp_jpeg_path = output_path + "_temp.jpg"
        original.save(temp_jpeg_path, 'JPEG', quality=quality)

        # Step 2: reopen recompressed image
        recompressed = Image.open(temp_jpeg_path).convert('RGB')

        # Step 3: pixel-wise difference
        ela_image = ImageChops.difference(original, recompressed)

        # Step 4: scale differences so artefacts become visible
        extrema = ela_image.getextrema()
        max_diff = max(ex[1] for ex in extrema)
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        # Optional: resize to model input size for nicer display
        ela_image = ela_image.resize((IMG_SIZE, IMG_SIZE))

        # Save final ELA image (PNG so no extra JPEG compression)
        ela_image.save(output_path, 'PNG')

        # Clean temp file
        if os.path.exists(temp_jpeg_path):
            os.remove(temp_jpeg_path)

        logger.info(f"ELA image saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"ELA generation failed: {str(e)}")
        return False

def generate_gradcam(input_path, output_path, last_conv_layer_name="conv5_block3_out"):
    """
    Generate Grad-CAM heatmap overlay for the uploaded image.
    """
    try:
        img = Image.open(input_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img).astype(np.float32)
        x = preprocess_input(img_array)
        x = np.expand_dims(x, axis=0)

        # Grad-CAM model
        grad_model = models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            pred_index = tf.argmax(predictions[0])
            pred_output = predictions[:, pred_index]

        grads = tape.gradient(pred_output, conv_outputs)[0]     # (H, W, C)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))       # (C,)

        conv_outputs = conv_outputs[0]
        conv_outputs *= pooled_grads

        heatmap = tf.reduce_mean(conv_outputs, axis=-1).numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) == 0:
            return False
        heatmap /= np.max(heatmap)

        # Resize & colorize
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cm.get_cmap("jet")(heatmap)[:, :, :3] * 255
        heatmap = heatmap.astype(np.uint8)

        overlay = cv2.addWeighted(
            np.array(img).astype(np.uint8), 0.6,
            heatmap, 0.4, 0
        )
        overlay_img = Image.fromarray(overlay)
        overlay_img.save(output_path, 'PNG')

        logger.info(f"Grad-CAM image saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {str(e)}")
        return False

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route('/')
def index():
    """Render the home page with upload form"""
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            flash('No image file provided. Please upload an image.', 'error')
            return redirect(url_for('index'))
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            flash('No file selected. Please choose an image to upload.', 'error')
            return redirect(url_for('index'))
        
        # Validate file extension
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP images only.', 'error')
            return redirect(url_for('index'))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Image uploaded: {filename}")
        
        # Preprocess image
        try:
            preprocessed_image = preprocess_image(filepath)
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            flash('Error processing image. Please try another image.', 'error')
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for('index'))
        
        # Get prediction
        try:
            prediction, confidence = predict_image(preprocessed_image)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            flash('Error during prediction. Please try again.', 'error')
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for('index'))

        # ----- Generate ELA image for slider -----
        base_name = filename.rsplit('.', 1)[0]
        ela_filename = f"ela_{base_name}.png"
        ela_filepath = os.path.join(app.config['UPLOAD_FOLDER'], ela_filename)
        ela_ok = perform_ela(filepath, ela_filepath)
        if not ela_ok:
            ela_filename = None

        # ----- Generate Grad-CAM overlay -----
        gradcam_filename = f"gradcam_{base_name}.png"
        gradcam_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
        gradcam_ok = generate_gradcam(filepath, gradcam_filepath)
        if not gradcam_ok:
            gradcam_filename = None

        # Update history in session (last 6 items)
        history = session.get('history', [])
        history_item = {
            'filename': filename,
            'prediction': prediction,
            'confidence': confidence
        }
        history.insert(0, history_item)
        history = history[:6]
        session['history'] = history

        # Prepare result data
        result_data = {
            'prediction': prediction,
            'confidence': confidence,
            'image_filename': filename,
            'image_path': filepath,
            'ela_filename': ela_filename,
            'gradcam_filename': gradcam_filename,
            'history': history
        }
        
        logger.info(f"Prediction complete: {prediction} ({confidence}% confidence)")
        
        return render_template('result.html', **result_data)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        flash('An unexpected error occurred. Please try again.', 'error')
        return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    logger.info("Starting Fake Image Detection System...")
    app.run(debug=True, host='0.0.0.0', port=5000)
