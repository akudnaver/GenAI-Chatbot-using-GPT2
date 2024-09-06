# Domain Knowledge Expert Chatbot

Welcome to the Domain Knowledge Expert Chatbot project! This repository contains code and resources for creating a chatbot that acts as a domain knowledge expert. The chatbot is trained using a GPT-2 pre-trained model and is capable of understanding and responding to user queries based on the provided input PDF.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Running the Chatbot](#running-the-chatbot)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This chatbot is designed to assist users by providing expert knowledge as fed by an input pdf, it could be domain specific data or any other data of your choice that you would like to interact with. By leveraging the GPT-2 pre-trained model, the chatbot can understand complex queries and provide accurate responses based on the content of an input PDF document.

## Features

- **GPT-2 Integration**: Utilizes the GPT-2 pre-trained model for natural language understanding and generation.
- **PDF Parsing**: Extracts content from input PDF documents to build a knowledge base for the chatbot.
- **Domain-Specific Expertise**: Trained specifically for telecom-related queries.
- **Interactive Interface**: Provides a user-friendly interface for interacting with the chatbot using Gradio.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/akudnaver/telecom-chatbot.git
    cd chatbot
    ```

2. **Create a virtual environment and activate it**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare the Input PDF**:
    - Place your telecom domain-related PDF document in the `input` directory.

2. **Parse the PDF and Train the Model**:
    - Run the script to parse the PDF and prepare the training data:
      ```sh
      python parse_pdf.py --input_file=input/your_pdf_file.pdf
      ```
    Note:  Here you can modify the Code to parse argument through the CLI if you are using IDE like Visual Studio otherwise you can use the method to specify a filename at
           the begining of the code and that's about it !!!

4. **Train the Chatbot Model**:
    - Train the GPT-2 model with the prepared data:
      ```sh
      python train_model.py --data_file=output/parsed_data.txt
      ```

5. **Run the Chatbot**:
    - Start the chatbot server:
      ```sh
      python app.py
      ```
    - Access the chatbot interface through Gradio.

## Training the Model

To train the GPT-2 model, you need to follow these steps:

1. **Prepare Training Data**:
    - The `parse_pdf.py` script extracts text from the input PDF and saves it as a plain text file.

  Note: This is all take care in the Code, so just refer the code !!!

2. **Train the GPT-2 Model**:
    - The `train_model.py` script uses the extracted text to fine-tune the GPT-2 model.

## Running the Chatbot

To interact with the chatbot, start the server by running the `app.py` script. The chatbot will be accessible through a Gradio interface.

## Contributing

We welcome contributions to enhance the capabilities and features of the Telecom Domain Knowledge Expert Chatbot. If you have any suggestions or improvements, please feel free to submit a pull request or open an issue.
