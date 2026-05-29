import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from scipy import ndimage

cnn_model = tf.keras.models.load_model('mnist_cnn_model.keras')

def preprocess(image):
    if isinstance(image, dict):
        image = image['composite']
    
    # RGBA → grayscale
    img = Image.fromarray(image.astype('uint8')).convert('L')
    
    # Invert (white bg → black bg like MNIST)
    img = ImageOps.invert(img)
    img_array = np.array(img)
    
    # Remove empty borders
    rows = np.any(img_array > 30, axis=1)
    cols = np.any(img_array > 30, axis=0)
    
    if not rows.any() or not cols.any():
        return None
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Crop to digit
    img_array = img_array[rmin:rmax+1, cmin:cmax+1]
    
    # Add padding
    img_array = np.pad(img_array, 4, mode='constant', constant_values=0)
    
    # Resize to 28x28
    img = Image.fromarray(img_array).resize((28, 28), Image.LANCZOS)
    img_array = np.array(img)
    
    # Center using center of mass
    cy, cx = ndimage.center_of_mass(img_array)
    shift_y = int(14 - cy)
    shift_x = int(14 - cx)
    img_array = ndimage.shift(img_array, [shift_y, shift_x])
    
    return img_array

def predict_digit(image):
    img_array = preprocess(image)
    
    if img_array is None:
        return {"Draw a digit first": 1.0}
    
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    prediction = cnn_model.predict(img_array, verbose=0)
    return {str(i): float(prediction[0][i]) for i in range(10)}

app = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(
        type='numpy',
        brush=gr.Brush(
            default_size=20,
            default_color="#000000",
            colors=["#000000", "#E24B4A", "#378ADD", "#1D9E75"],
            color_mode="select"
        )
    ),
    outputs=gr.Label(num_top_classes=3),
    title='Handwritten Digit Recognizer',
    description='Draw any digit clearly and thickly!'
)

app.launch(share=True)