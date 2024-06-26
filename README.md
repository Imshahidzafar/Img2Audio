# Image to Audio Story Converter

This Streamlit application takes an image as input, extracts text from it, generates a short story based on the extracted text, and finally converts the story into audio. It leverages models from Huggingface and OpenAI to process the image and text.

## Features

- **Image Text Extraction:** Uses Salesforce's BLIP model for image captioning.
- **Story Generation:** Utilizes OpenAI's GPT-3.5 model to generate short narratives based on the image text.
- **Text to Speech:** Converts the generated story to audio using the espnet/kan-bayashi_ljspeech_vits model from Huggingface.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Imshahidzafar/Img2Audio
   cd Img2Audio
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your API keys in .env file:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
   ```
## Usage
To run the application:
   ```bash
   streamlit run app.py
   ```
   
Navigate to http://localhost:8501 in your web browser to view the app. Upload an image file, and the app will display the extracted text, the generated story, and the audio playback of the story.

## Models Used
- **Image-to-Text**: Salesforce/blip-image-captioning-base
- **Story Generation**: OpenAI's GPT-3.5-turbo
- **Text-to-Speech**: espnet/kan-bayashi_ljspeech_vits 
 
