# Fake Image Detection System

A professional full-stack Flask web application for detecting and classifying fake digital images using deep learning.

## 🚀 Features

- **Modern UI**: Clean, responsive design with Bootstrap 5 and gradient animations
- **Image Upload**: Support for PNG, JPG, JPEG, GIF, and BMP formats
- **Image Preprocessing**: Automatic grayscale conversion, resizing (128x128), and normalization
- **AI Prediction**: Deep learning model analysis with confidence scores
- **Real-time Results**: Instant feedback with detailed analysis results

## 📋 Requirements

- Python 3.8 or higher
- pip (Python package manager)

## 🛠️ Installation

1. **Navigate to the project directory:**
   ```bash
   cd fake_image_detector
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

1. **Start the Flask application:**
   ```bash
   python app.py
   ```

2. **Open your web browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Upload an image:**
   - Click "Select Image" and choose an image file
   - Preview will appear automatically
   - Click "Detect Image Authenticity" to analyze

4. **View results:**
   - See prediction (Real/Fake)
   - Check confidence score
   - View analysis details

## 📁 Project Structure

```
fake_image_detector/
├── app.py                 # Flask backend application
├── model/                 # Model directory (for future CNN model)
│   └── fake_image_model.h5
├── static/
│   ├── css/
│   │   └── style.css      # Custom styles
│   ├── js/
│   │   └── script.js      # Client-side JavaScript
│   └── images/            # Static images
├── templates/
│   ├── index.html         # Home page
│   └── result.html        # Results page
├── uploads/               # Uploaded images (auto-created)
└── requirements.txt       # Python dependencies
```

## 🔧 Configuration

- **Max file size**: 16MB (configurable in `app.py`)
- **Allowed formats**: PNG, JPG, JPEG, GIF, BMP
- **Image preprocessing**: 128x128 grayscale, normalized (0-1)
- **Port**: 5000 (default Flask port)

## 🧠 Model Integration

Currently, the system uses a placeholder prediction algorithm. To integrate your trained CNN model:

1. Save your trained Keras model as `model/fake_image_model.h5`
2. Update the `predict_image()` function in `app.py` to load and use the actual model:

```python
from tensorflow import keras

# Load model (add to app initialization)
model = keras.models.load_model('model/fake_image_model.h5')

# Update predict_image function
def predict_image(image_array):
    # Reshape for model input if needed
    prediction = model.predict(np.expand_dims(image_array, axis=0))
    # Process prediction and return result
    ...
```

## 🐛 Troubleshooting

- **Import errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
- **Port already in use**: Change the port in `app.py` (last line)
- **Image upload fails**: Check file size (max 16MB) and format (PNG, JPG, etc.)
- **Model not found**: The system works without a model file (uses placeholder prediction)

## 📝 Notes

- This is a demonstration system with placeholder prediction logic
- For production use, integrate a trained CNN model
- Results are based on pattern analysis and should be used as reference only
- Consult image forensics experts for critical applications

## 📄 License

This project is provided as-is for educational and demonstration purposes.

## 👨‍💻 Development

The application is built with:
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, Bootstrap 5
- **Image Processing**: OpenCV, Pillow
- **Deep Learning**: TensorFlow/Keras (ready for model integration)

---

**Ready to detect fake images!** 🎨🔍