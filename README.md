
# CNN-Based Counterfeit Currency Detection

## Project Overview
The CNN-Based Counterfeit Currency Detection system is a deep learning application designed to identify whether an Indian currency note is Real or Fake using a Convolutional Neural Network (CNN).

The system analyzes uploaded currency note images and predicts their authenticity through a user-friendly web application.

## Objectives
- To detect counterfeit Indian currency notes using deep learning.
- To classify currency notes as Real or Fake.
- To train a CNN model using currency note images.
- To develop a simple web application for currency detection.

## Technologies Used
- Programming Language: Python
- Deep Learning: Convolutional Neural Network (CNN)
- Libraries: TensorFlow, Keras, NumPy, Pillow
- Dataset: Kaggle Indian Currency Real and Fake Notes Dataset
- Development Platform: Google Colab
- Web Framework: Streamlit
- Deployment: GitHub and Streamlit Community Cloud

## Methodology
1. Data Collection: Collected real and fake Indian currency note images from Kaggle.
2. Data Preprocessing: Resized and prepared images for model training.
3. Model Building: Developed a CNN model to extract image features.
4. Model Training: Trained the model using the prepared dataset.
5. Prediction: Classified uploaded currency images as Real or Fake.
6. Web Application: Integrated the trained model with Streamlit.

## Model Performance
- Model: Convolutional Neural Network (CNN)
- Classification: Real and Fake
- Test Accuracy: 81.7%

## Features
- User-friendly web interface.
- Upload currency note images.
- Automatic image classification.
- Displays Real or Fake prediction results.
- Deep learning-based detection.

## How to Run the Project

1. Clone this repository.
2. Install the required dependencies:

   pip install -r requirements.txt

3. Run the Streamlit application:

   streamlit run app.py

4. Upload an Indian currency note image.
5. View the prediction result.

## Future Enhancements
- Improve model accuracy using a larger dataset.
- Support multiple Indian currency denominations.
- Enhance counterfeit detection performance.
- Introduce real-time currency detection.

## Conclusion
This project demonstrates how deep learning and CNN can be used for counterfeit currency detection. It provides a simple web-based solution to classify Indian currency notes as Real or Fake.

## Domain
Artificial Intelligence and Deep Learning
