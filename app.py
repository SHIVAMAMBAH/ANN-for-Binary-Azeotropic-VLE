import streamlit as st
import onnxruntime as ort
import numpy as np

# Load ONNX model
model_path = "ann_model.onnx"
session = ort.InferenceSession(model_path)

# Get the first input name (usually only one input tensor)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

st.title("ANN Model for Binary Azeotropic VLE System")
st.write("Enter values for the model inputs:")

# Streamlit input fields for 3 features
input1 = st.number_input("x1_mol", value=0.0)
input2 = st.number_input("T_k", value=0.0)
input3 = st.number_input("P_atm", value=0.0)

# Button to make prediction
if st.button("Predict"):
    # Prepare input as numpy array of shape (1, 3)
    input_array = np.array([[input1, input2, input3]], dtype=np.float32)
    
    # Run the model
    prediction = session.run([output_name], {input_name: input_array})[0]

    st.success(f"Predicted Output (y1_mol): {prediction[0][0]}")
