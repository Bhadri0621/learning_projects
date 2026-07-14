import streamlit as st 
from model import LSTMModel,VanillaRNN,GRU
from preprocessing import preprocessing
from inference import prediction 
models=['lstm','gru','rnn']
input=st.text_input('Enter a sentence ')
model=st.selectbox(label='select one model',options=['LSTM','RNN','GRU'])
if model:
    if st.button('predict'):
        # with st.spinner:
            inp=preprocessing(input)
            topk,arg=prediction(inp,model)
            # print(f'Topk result :: {topk}')
            # print(f'Argmax :: {arg}')
            st.write(topk)
            st.write(arg)
