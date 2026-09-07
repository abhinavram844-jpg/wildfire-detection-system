# Wildfire Detection System

An AI-based wildfire detection system that uses deep learning to classify images as smoke or no-smoke

## Features

- Image classification using a fine-tuned ResNet-18 model
- Dataset preparation and train/validation/test splitting
- Model training and evaluation
- Image prediction with confidence scores
- SMS alert functionality using Twilio
- Designed as part of a larger wildfire detection system involving Raspberry Pi hardware and environmental sensors


## Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet-18
- Scikit-learn
- Twilio
- Raspberry Pi

## Project Structure

```text
wildfire-detection/
├── src/
│   ├── wildfire_train.py
│   ├── predict.py
│   ├── split_dataset.py
│   └── SMS.py
├── models/
├── requirements.txt
└── .gitignore
```

## Machine Learning Model

The project uses transfer learning with a pretrained ResNet-18 convolutional neural network. The model was adapted for binary image classification to distinguish between smoke and non-smoke images.

During training, most of the pretrained network was frozen while later layers and the final classification layer were fine-tuned for the wildfire detection task.

## Future Work

- Integrate real-time camera input 
- Deploy the model on raspberry pi hardware
- Integrate environmental sensors
- Trigger automated emergency alerts based on detection results  


