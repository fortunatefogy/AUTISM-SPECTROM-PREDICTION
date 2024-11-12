import streamlit as st
import pandas as pd
import pickle


@st.cache_data
def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title('Autism Spectrum Disorder Prediction')
    st.write('Enter the required information and click on Predict.')

    
    st.sidebar.header('Input Parameters')

    df = pd.read_csv("data/Autisms_data.csv")  

    st.sidebar.subheader('Sample Input Data')
    st.sidebar.write(df.head())

 
    A1_Score = st.sidebar.selectbox('A1 Score', df['A1_Score'].unique())
    A2_Score = st.sidebar.selectbox('A2 Score', df['A2_Score'].unique())
    A3_Score = st.sidebar.selectbox('A3 Score', df['A3_Score'].unique())
    A4_Score = st.sidebar.selectbox('A4 Score', df['A4_Score'].unique())
    A5_Score = st.sidebar.selectbox('A5 Score', df['A5_Score'].unique())
    A6_Score = st.sidebar.selectbox('A6 Score', df['A6_Score'].unique())
    A7_Score = st.sidebar.selectbox('A7 Score', df['A7_Score'].unique())
    A8_Score = st.sidebar.selectbox('A8 Score', df['A8_Score'].unique())
    A9_Score = st.sidebar.selectbox('A9 Score', df['A9_Score'].unique())
    A10_Score = st.sidebar.selectbox('A10 Score', df['A10_Score'].unique())
    age = st.sidebar.number_input('Age', min_value=1, max_value=100, value=18)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    jaundice = st.sidebar.selectbox('Jaundice', ['Yes', 'No'])
    autism = st.sidebar.selectbox('Autism', ['Yes', 'No'])
    relation = st.sidebar.selectbox('Relation', df['relation'].unique())

  
    input_data = pd.DataFrame({
        'A1_Score': [A1_Score],
        'A2_Score': [A2_Score],
        'A3_Score': [A3_Score],
        'A4_Score': [A4_Score],
        'A5_Score': [A5_Score],
        'A6_Score': [A6_Score],
        'A7_Score': [A7_Score],
        'A8_Score': [A8_Score],
        'A9_Score': [A9_Score],
        'A10_Score': [A10_Score],
        'age': [age],
        'gender': [1 if gender == 'Male' else 0],  
        'jaundice': [1 if jaundice == 'Yes' else 0],
        'austim': [1 if autism == 'Yes' else 0],
        'relation': [relation]
    })


    model = load_model('logistic_regression_model.pkl') 


    if input_data.shape[1] != 15:  
        st.write(f"Error: The model expects 15 features, but got {input_data.shape[1]}. Please check your inputs.")
        return
    
    st.markdown(
        """
        <style>
        body {
            background-image: url('image.jpg');  
            background-size: cover;
            background-repeat: no-repeat;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    
    if st.sidebar.button('Predict'):
        prediction = predict(model, input_data)
        if prediction == 1:
            st.write('Prediction: Autism Spectrum Disorder')
        else:
            st.write('Prediction: Not Autism Spectrum Disorder')

if __name__ == '__main__':
    main()
