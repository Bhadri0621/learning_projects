from inference import Prediction
import streamlit as st
from PIL import Image
from torchvision import transforms
from classes import classes,descriptionn
st.markdown("""
<style>
.stApp {
    background-color: #F1F8E9;
}
</style>
""", unsafe_allow_html=True)
photo=Image.open('plant_image.jpg')
banner=photo.crop((0,500,photo.width,1200)) 
st.image(banner,use_container_width=True)
st.title('🌿 Plant Disease Detection ')
st.caption('AI-Powered Crop Health Analysis')
# st.markdown('Hello Farmer what happened!')
input_file=st.file_uploader('Upload image',type=['jpg','jpeg','png'])
if input_file is not None:
  if st.button('Predict'):
    with st.spinner('Predicting..'):
      image=Image.open(input_file).convert('RGB')
      transform=transforms.Compose([
          transforms.Resize((128,128)),
          transforms.ToTensor(),
          transforms.Normalize(
            mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5])
        ])
      image=transform(image)
      image=image.unsqueeze(0)
      prediction,class_name=Prediction(image)
      des_mess=descriptionn[class_name]['description']
      Treat=descriptionn[class_name]['Treatment']

    col1,col2=st.columns(2)
    with col1:
      st.image(input_file,use_container_width=True)

    with col2:
      st.success(f'Disease :: {class_name}')
      st.success(f'Confidence :: {prediction}')
      st.subheader('Description')
      st.write(des_mess)
      # st.success(f'Description :: {des_mess}')
      # st.success(f'Treatment :: {Treat}') 
      st.subheader('Treatment')
      st.write(Treat)

