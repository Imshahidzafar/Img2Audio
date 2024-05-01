import os
import requests
from dotenv import load_dotenv
from transformers import pipeline
import streamlit as st

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

def img2txt(url):
    image2txt = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", max_new_tokens=100)
    text = image2txt(url)[0]['generated_text']
    print(text)
    return text

def generate_story(scenario):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [{"role": "system", "content": """
                    You are a story teller:
                    You can generate a short story based on a simple narrative, the story should be no more than 20 words:
                    """}, 
                     {"role": "user", "content": scenario}]
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    if response.status_code == 200:
        story = response.json()['choices'][0]['message']['content'].strip()
        print(story)
        return story
    else:
        print("Error:", response.json())

# txt2speech
def txt2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": message
    }
    
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac','wb') as file:
        file.write(response.content)

# scenario = img2txt("photo.jpg")
# story = generate_story(scenario)
# txt2speech(story)

def main():
    st.set_page_config(page_title="img 2 audio story",page_icon="👨‍💻")
    st.header("Turn img into audio story")
    uploaded_file= st.file_uploader("Choose an image...",type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name,"wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file,caption="Uploaded Image.",use_column_width=True)
        scenario = img2txt(uploaded_file.name)
        story = generate_story(scenario)
        txt2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        st.audio("audio.flac")

if __name__=='__main__':
    main()