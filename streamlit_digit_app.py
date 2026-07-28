
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Configure the Streamlit page (browser tab title, icon, and centered layout)
st.set_page_config(page_title='Handwritten Digit Recognition', page_icon='🔢', layout='centered')

# Cache the model so it's loaded from disk only once per session,
# not re-loaded every time the app reruns (e.g. on button click)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_recognizer.keras")

model = load_model()

# App title and instructions
st.title('🔢 Handwritten Digit Recognition')
st.write('Draw a digit (0-9) below and click Predict.')

# Drawable canvas widget where the user sketches a digit
# - white strokes on a black background mimic MNIST's image style
# - 280x280 gives a large drawing area, later downscaled to 28x28 for the model
canvas_result = st_canvas(
    fill_color='blue',
    stroke_width=15,
    stroke_color='white',
    background_color='black',
    width=280,
    height=280,
    drawing_mode='freedraw',
    key='canvas',
)

# Only run prediction logic when the user clicks the "Predict" button
if st.button('Predict'):
    if canvas_result.image_data is not None:

        # Convert the canvas's raw pixel array (RGBA) into a PIL Image
        img = Image.fromarray(canvas_result.image_data.astype('uint8'))

        # Convert to grayscale ('L' mode) since the model expects single-channel input
        img = img.convert('L')

        # Resize down to 28x28 to match the MNIST input size
        img = img.resize((28, 28))

        img = np.array(img)               # convert to numpy array for numerical processing
        img = img / 255.0                 # normalize pixel values to [0, 1] range
        img = img.reshape(1, 28, 28, 1)   # reshape to (1, 28, 28, 1): batch of 1, single channel

        # Run the model on the preprocessed image
        prediction = model.predict(img, verbose=0)

        # predicted digit is the class with the highest probability
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        # Display the results
        st.success(f'Predicted Digit: {digit}')
        st.metric('Confidence', f'{confidence * 100:.2f}%')
        st.subheader('Prediction Probabilities')
        st.bar_chart(prediction[0])

    else:                                  # when no drawing is detected
        st.warning('Please draw a digit first.')
