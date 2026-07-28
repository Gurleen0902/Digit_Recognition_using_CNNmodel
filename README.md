# Handwritten Digit Recognition — CNN + Streamlit

A Convolutional Neural Network trained on the MNIST dataset to recognize handwritten digits (0–9), deployed as an interactive Streamlit web app where users can draw a digit and get a real-time prediction.

**Built as part of the IBM AI Training Program (120 hours).**

## Results

- **Test Accuracy:** 99.37%
- **Test Loss:** 0.0235

##  Model Architecture
Conv2D(32, 3x3, relu) → MaxPooling2D(2x2)
Conv2D(64, 3x3, relu) → MaxPooling2D(2x2)
Flatten → Dense(64, relu) → Dropout(0.5)
Dense(10, softmax)

- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Trained for 10 epochs on 60,000 MNIST training images, validated on 10,000 test images

##  Demo

Draw a digit on the canvas → click **Predict** → see the predicted digit, confidence score, and full probability distribution across all 10 classes.

## Repository Contents

- `CNN-handwritten-digit-recognition.ipynb` — full training notebook (data loading, preprocessing, model building, training, evaluation)
- `CNN_Model.ipynb` — model development/experimentation notebook
- `digit_recognizer.keras` — trained model weights
- `streamlit_digit_app.py` — Streamlit web app for interactive predictions
- `Digit_recog_project_CNN_byGurleenKaur.pptx` — project presentation

##  Tech Stack

- **TensorFlow / Keras** — model building and training
- **NumPy, Matplotlib** — data processing and visualization
- **Streamlit** — web app deployment
- **streamlit-drawable-canvas** — interactive drawing input

##  Run Locally

```bash
git clone https://github.com/Gurleen0902/handwritten-digit-recognition-cnn.git
cd handwritten-digit-recognition-cnn
pip install -r requirements.txt
streamlit run streamlit_digit_app.py
```

##  Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) — 70,000 grayscale images (28x28) of handwritten digits, loaded via `tensorflow.keras.datasets.mnist`.

## 👤 Author

**Gurleen Kaur** — BCA student, GGDSD College Chandigarh
