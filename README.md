# Rainforest Audio Classifier

> Empowering Conservation Through AI on the Edge 🌿

This project focuses on classifying audio signals into two categories: **environmental sounds** and **chainsaw activity**, by using a scaled-down machine learning model, aiding in monitoring and combating illegal deforestation activities in rainforest areas.

The trained model is designed for deployment on an **ESP32** microcontroller for edge computing applications.

![esp32](assets/esp32.jpg)

## Dataset Specification

The dataset comprises a large collection of short audio clips capturing sounds of chainsaws at varying distances, alongside natural environmental sounds. These recordings were gathered using **Guardian devices** deployed by the [Rainforest Connection (RFCx)](https://rfcx.org/), a non-profit organization leveraging eco-acoustics and AI to protect endangered ecosystems.

### Key Characteristics

- **Regions Covered:** Most recordings originate from _South America_ and _Southeast Asia_.
- **Objective:** This dataset is specifically curated to train AI models for detecting chainsaw activity, supporting efforts to combat illegal logging in rainforests.
- **Audio format:** `.wav` files.
- **Class Labels:**
  ```json
  { "0": "chainsaw", "1": "environment" }
  ```
- **Dataset Split:** The dataset has been divided into a 70:30 train-test ratio by the organization.
  - Train dataset: `35275` non-corrupt audio files.
  - Test dataset: `15117` non-corrupt audio files.

### Dataset Source

The data is publicly available on the [Hugging Face Datasets Repository](https://huggingface.co/datasets/rfcx/frugalai) as part of the [FrugalAI Challenge](https://frugalaichallenge.org/).

### Sample Audio Files

#### Environment:

<audio controls="controls">
  <source type="audio/mp3" src="assets/env.mp3"></source>
  <p>Your browser does not support the audio element. <a href="assets/env.mp3" target="_blank">View in new tab.</a></p>
</audio>

#### Chainsaw:

<audio controls="controls">
  <source type="audio/mp3" src="assets/chainsaw.mp3"></source>
  <p>Your browser does not support the audio element. <a href="assets/chainsaw.mp3" target="_blank">View in new tab.</a></p>
</audio>
