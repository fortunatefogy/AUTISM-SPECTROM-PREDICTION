import streamlit as st
import pandas as pd
import pickle

# Load the trained model
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

    
    # st.image('image.jpg', use_column_width=True)
    st.sidebar.header('Input Parameters')

    
    df = pd.read_csv("data/Autisms_data.csv")  

    
    #st.sidebar.subheader('Sample Input Data')
    #st.sidebar.write(df.head())

   
    A1_Score = st.sidebar.selectbox('often notice small sounds when others do not.', df['A1_Score'].unique())
    A2_Score = st.sidebar.selectbox('concentrate more on the whole picture,rather than the small details.', df['A2_Score'].unique())
    A3_Score = st.sidebar.selectbox('easy to do more than one thing at once.', df['A3_Score'].unique())
    A4_Score = st.sidebar.selectbox('If there is an interruption, I can switch back to what I was doing very quickly.', df['A4_Score'].unique())
    A5_Score = st.sidebar.selectbox('find it easy to read between the lines when someone is talking to me', df['A5_Score'].unique())
    A6_Score = st.sidebar.selectbox('I know how to tell if someone listening to me is getting bored.', df['A6_Score'].unique())
    A7_Score = st.sidebar.selectbox('When I’m reading a story, I find it difficult to work out the character’s intention', df['A7_Score'].unique())
    A8_Score = st.sidebar.selectbox('I like to collect information about categories of things (', df['A8_Score'].unique())
    A9_Score = st.sidebar.selectbox(': I find it easy to work out what someone is thinking or feeling just by looking at their face.', df['A9_Score'].unique())
    A10_Score = st.sidebar.selectbox('I find it difficult to work out people’s intentions', df['A10_Score'].unique())
    age = st.sidebar.number_input('Age', min_value=1, max_value=100, value=18)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    jaundice = st.sidebar.selectbox('Jaundice at the time of birth', ['Yes', 'No'])
    autism = st.sidebar.selectbox('immediate family member has been diagnosed with autism', ['Yes', 'No'])
    relation = st.sidebar.selectbox('Relation of the patient who completed the test', df['relation'].unique())

    # Convert input into a DataFrame
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

    # Load model
    model = load_model('logistic_regression_model.pkl')  

    # Ensure input data has the same number of features as the training data
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


    # Predict button
    if st.sidebar.button('Predict'):
        prediction = predict(model, input_data)
        print("prediction:",prediction)
        if prediction == 1:
            st.write('Prediction: Autism Spectrum Disorder')
        else:
            st.write('Prediction: Not Autism Spectrum Disorder')

if __name__ == '__main__':
    main()
