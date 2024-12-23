import streamlit as st
import numpy as np
import json
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import load_model

# Load the model
model = load_model(r'D:\AI Course\final project\Chatbot\Chatbot\SaveModels\.ipynb_checkpoints\model.h5')

# Load tokenizer
with open(r'D:\AI Course\final project\Chatbot\Chatbot\SaveModels\.ipynb_checkpoints\toknizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

label_encoder = pickle.load(open(r'D:\AI Course\final project\Chatbot\Chatbot\SaveModels\.ipynb_checkpoints\label_encoder.pkl', 'rb'))

# Load intents data
with open('../intents.json') as file:
    data = json.load(file)


def predict(value):
    # Preprocess the input text
    sequence = pad_sequences(tokenizer.texts_to_sequences([value]), maxlen=20)   
    # Make predictions
    result = np.argmax(model.predict(np.array(sequence)))

    # Inverse transform the label
    f_res = label_encoder.inverse_transform(np.array(result).reshape(1))

    output = ''
    for label in data['intents']:
        if label['tag'] == f_res:
            output = np.random.choice(label['responses'])

    return output


# Streamlit app
def main():
    st.title("College Chatbot")

    # Input text box for user input
    user_input = st.text_input("Enter your text:")
    
    # Predict button
    if st.button("Predict"):
        if user_input:
            prediction = predict(user_input)
            st.write(f"Chatbot: {prediction}")
        else:
            st.warning("Please enter some text for prediction")


if __name__ == "__main__":
    main()
