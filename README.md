# Building a Hybrid Chatbot with Multimodal Capabilities and LLMs for Hotel Recommendations

## Contents


1. Introduction
2. Understanding Multimodal Systems
3. Differences Between Multimodal Systems and Conventional LLMs
4. Project Overview: Building the Hotel Recommendation Chatbot
5. Explanation of Each Function in the Code
6. Front-End Application Implementation
7. Conclusion
8. Summary

## 1. Introduction

### How to Create a Hybrid Chatbot: Integrating Multimodal Systems with Large Language Models


In the modern world, chatbots are becoming increasingly sophisticated, thanks to advancements in machine learning and artificial intelligence. By leveraging large language models (LLMs) and multimodal capabilities, we can create hybrid chatbots that not only understand text but also analyze images, audio, and more. In this blog post, we will guide you through building a hybrid chatbot that provides hotel recommendations. This chatbot will utilize both LLMs and multimodal processing to deliver rich and informative responses, enhancing the user experience.

## 2. Understanding Multimodal Systems

A multimodal system is a type of AI that can process and generate data across multiple modalities, such as text, images, audio, and video. Unlike traditional models that are limited to one form of data (e.g., text-only or image-only), multimodal systems can understand and synthesize information from different sources. This makes them particularly powerful for applications that require a more holistic understanding of the world, such as visual question answering, image captioning, and in our case, hotel recommendations.

## 3. Differences Between Multimodal Systems and Conventional LLMs

### Multimodal Systems
- **Data Inputs:** Can process multiple forms of data, such as images, text, and even audio.
- **Output:** Provides richer responses that may include text descriptions, analyzed images, or synthesized audio responses.
- **Applications:** Useful in areas like image captioning, visual question answering, and any task requiring multi-sensory input.

### Conventional LLMs
- **Data Inputs:** Primarily processes text data.
- **Output:** Generates text-based responses.
- **Applications:** Ideal for text generation, translation, summarization, and conversational agents.

In our project, the combination of these two approaches allows us to create a more dynamic and contextually aware chatbot.

## 4. Project Overview: Building the Hotel Recommendation Chatbot

We will build a hybrid chatbot that, when given a place, finds the nearest hotels and analyzes their images using multimodal processing. It will then append the results and analyze them using a conventional LLM. The end goal is to provide users with hotel recommendations based on both the proximity to the location and the visual appeal of the hotel, as analyzed by our multimodal system.

### Project Structure:
- **Multimodal Analysis:** Analyze hotel images to assess their quality and appeal.
- **LLM Integration:** Use a text-based LLM to generate recommendations based on the analyzed data.
- **Frontend Application:** Create a simple and user-friendly interface for users to interact with the chatbot.

## 5. Explanation of Each Function in the Code

### Importing Libraries
```python
import gradio as gr
from gradio_multimodalchatbot import MultimodalChatbot
from gradio.data_classes import FileData
import os
import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import matplotlib.pyplot as plt
import urllib3
from transformers import pipeline
from transformers import BitsAndBytesConfig
import torch
import textwrap
from haversine import haversine  # Install haversine library: pip install haversine
from transformers import AutoProcessor, LlavaForConditionalGeneration
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoImageProcessor
from datasets import load_dataset
from geopy.geocoders import Nominatim
```
These imports include libraries for handling image processing, geolocation, data handling, and model inference.

### Setting Up the Environment
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOW_MEMORY = os.getenv("LOW_MEMORY", "0") == "1"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
TEXT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
```
This section sets up the device and model IDs, ensuring that the code runs efficiently.

### Loading the Models
```python
# Load the tokenizer, image processor, and models
tokenizer_image_to_text = AutoTokenizer.from_pretrained(MODEL_ID)
image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, quantization_config=quantization_config, device_map="auto")
pipe_image_to_text = pipeline("image-to-text", model=model, tokenizer=tokenizer_image_to_text, image_processor=image_processor, model_kwargs={"quantization_config": quantization_config})
pipe_text = pipeline("text-generation", model=TEXT_MODEL_ID, model_kwargs={"quantization_config": quantization_config, "use_auth_token": True})
```
Here, we initialize the models and pipelines necessary for multimodal processing and text generation.

### Handling Geolocation
```python
def get_coordinates(location_name):
    geolocator = Nominatim(user_agent="coordinate_finder")
    location = geolocator.geocode(location_name)
    if location:
        return location.latitude, location.longitude
    else:
        return None
```
This function gets the coordinates for a given location name using the Geopy library.

### Finding Nearby Hotels
```python
def find_nearby(place=None):
    ...
    closest_hotels = geocoded_hotels.sort_values(by='distance_km').head(5)
    return closest_hotels
```
This function finds hotels near the specified location by calculating the distance between the given location and hotel coordinates.

### Analyzing Hotel Images
```python
def search_hotel(place=None):
    ...
    for index, row in grouped_df.iterrows():
        ...
        description = outputs[0]["generated_text"].split("\nASSISTANT:")[-1].strip()
        description_data.append({'hotel_name': hotel_name, 'hotel_id': hotel_id, 'image': img, 'description': description})
    return pd.DataFrame(description_data)
```
This function retrieves hotel images and analyzes them using the multimodal pipeline, generating descriptive summaries of the hotels.

### Generating Text-Based Recommendations
```python
def generate_text_response(prompt):
    outputs = pipe_text(prompt, max_new_tokens=500)
    response = outputs[0]['generated_text'].split("[/INST]")[-1].strip()
    return response
```
This function generates text-based recommendations by feeding the analyzed data into the LLM.

### Creating the Chatbot Conversation
```python
def chatbot_response(user_input, conversation):
    ...
    hotel_conversation = multimodal_results(description_df)
    ...
    final_recommendation = llm_results(description_df)
    ...
```
This function manages the overall conversation flow, combining the results from both the multimodal and LLM pipelines.

## 6. Front-End Application Implementation

We use the Gradio library to create an interactive web interface for the chatbot.

```python
with gr.Blocks() as demo:
    gr.Markdown("# 🏨 Hotel Recommendation Chatbot")
    gr.Markdown("**Provide the location to discover hotels and receive personalized recommendations!**")

    initial_conv = initial_conversation()
    chatbot = MultimodalChatbot(value=initial_conv, height=500)

    with gr.Row():
        place_input = gr.Textbox(label="Enter a place", placeholder="E.g., Paris France, Tokyo Japan, Genova Italy")
        send_btn = gr.Button("Search Hotels")

    send_btn.click(chatbot_response, inputs=[place_input, chatbot], outputs=chatbot)

demo.launch(debug=True)
```
This code sets up the front-end, allowing users to input a location and receive recommendations directly in the browser.

## 7. Conclusion

In this blog post, we walked through the creation of a hybrid chatbot that utilizes both multimodal systems and LLMs to provide hotel recommendations. By integrating image analysis with text-based responses, we demonstrated how powerful and flexible modern AI systems can be. This project not only showcases the capabilities of multimodal and LLM-based systems but also provides a practical application that could be extended to various other domains, such as real estate, tourism, and e-commerce.

## 8. Summary

This blog covered the following key points:
- **Introduction:** Overview of hybrid chatbots using multimodal and LLM capabilities.
- **Multimodal Systems:** Explanation of multimodal systems and their advantages.
- **Project Overview:** Detailed walkthrough of building a hotel recommendation chatbot.
- **Code Explanation:** Step-by-step guide to the codebase used in the project.
- **Frontend Application:** Implementation of a user interface with Gradio.
- **Conclusion:** Recap of the project and its potential applications.

This project serves as a foundation for more advanced chatbot applications, merging the best of both multimodal and LLM technologies.
