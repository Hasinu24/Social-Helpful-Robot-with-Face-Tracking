# 🤖 Emotion Detecting Social Robot Using Facial and Voice Recognition with Artificial Intelligence (Face Tracking)

**Author:** Hasinu Ravishka

---

> A low-cost, Raspberry Pi-based social robot that recognises facial emotions in real time, engages in empathetic voice conversation, tracks faces, performs emotional gestures, and uniquely detects mismatches between what a person says and how they feel.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Hardware](#hardware)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Dataset Instructions](#dataset-instructions)
- [Running in Google Colab](#running-in-google-colab)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project presents an affordable, emotionally intelligent social robot designed for real-world companion applications — including **elderly care support**, **autism education**, and **basic mental health interaction**. Built around a Raspberry Pi 5 and a custom-trained PyTorch CNN, the robot bridges the gap between high-cost commercial social robots (£8,000–£15,000) and truly accessible, community-deployable alternatives — at a total build cost of approximately **£150**.

---

## ✨ Features

### 🧠 7-Class Real-Time Facial Emotion Recognition
- Custom PyTorch CNN trained on the **FER-2013** dataset
- Recognises: `Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`
- ~**95% validation accuracy** | ~**45 ms** inference on Raspberry Pi

### 👁️ Face Tracking & Emotional Gestures
- Pan servo continuously follows the user's face
- Context-aware gestures: waving on `Happy`, calming responses on `Sad`

### 🎙️ Voice-Based Natural Conversation
- **OpenAI Whisper** — accurate, offline-capable speech-to-text
- **GPT-4o-turbo** — emotion-conditioned, empathetic response generation
- **pyttsx3 TTS** — adaptive tone (slow and calm for sadness, energetic for happiness)

### 🔍 Novel Emotion–Voice Mismatch Detection
Detects conflicts between facial expression and spoken content (e.g. smiling while saying *"I'm sad"*) and responds with nuanced, empathetic acknowledgement — a novel feature not commonly found in low-cost social robots.

### 💾 Persistent Conversation Memory
Saves and reloads full chat history from JSON, maintaining meaningful context across multiple sessions.

---

## 🔧 Hardware

| Component | Details |
|---|---|
| **Microcontroller** | Raspberry Pi 5 (4 GB) |
| **Camera** | Logitech C270 USB Webcam |
| **Servos** | 3 × MG90S Micro Servos (pan + gesture) |
| **Audio** | USB Microphone & Speaker |
| **Total Cost** | ~£150 |

GPIO pin assignments: Pan servo → **GPIO 18**, Gesture servos → **GPIO 19** & **GPIO 15**

---

## 📁 Repository Structure

```
Social-Helpful-Robot-with-Face-Tracking/
│
├── Project_main_code.py                 # Main robot script: face detection, emotion
│                                        # recognition, voice loop, servo control
│
├── model_3.pth                          # Trained CNN weights (7 emotion classes)
│
├── CNN_model_Finetune.zip               # Colab notebook + files used for fine-tuning
│
├── FYP_GanttChart_Visual.xlsx           # Project timeline & Gantt chart
│
├── conversation_memory.json             # Example saved conversation history
│
├── haarcascade_frontalface_default.xml  # OpenCV Haar Cascade for face detection
│
└── voice_recognition_research.zip      # Research notes on voice processing
```

---

## 🚀 Getting Started

### Prerequisites

- Raspberry Pi 5 with camera and servos connected (GPIO 18 / 19 / 15)
- USB microphone and speaker
- Internet connection (required for OpenAI API calls)
- Python 3.9+

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Hasinu24/Social-Helpful-Robot-with-Face-Tracking.git
cd Social-Helpful-Robot-with-Face-Tracking
```

**2. Install dependencies**
```bash
pip install opencv-python torch torchvision torchaudio openai pyttsx3 gpiozero sounddevice numpy
```

**3. Add your OpenAI API key**

Either set it as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```
Or add it directly in `Project_main_code.py` where indicated.

**4. Ensure the model file is in the project root**
```
model_3.pth  ← should be here
```

**5. Run the robot**
```bash
python Project_main_code.py
```

### Controls

| Key | Action |
|-----|--------|
| `v` | Start voice conversation (say `"stop"` to end) |
| `p` | Object recognition mode (optional) |
| `q` | Quit |

---

## 🗂️ Dataset Instructions

> ⚠️ **Note:** The dataset is **not included** in this repository due to size limitations (>25 MB).

**1. Download the FER2013 dataset from Kaggle:**
👉 [FER2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

**2. Save the downloaded file as:**
```
images.zip
```

**3. Upload `images.zip`** to Google Colab or your local working directory.

**4. Extract the dataset:**
```python
!unzip images.zip -d ./dataset/
```

**5. Confirm the images are now located at:**
```
./dataset/
```

---

## ☁️ Running in Google Colab

**1.** Open Google Colab: [https://colab.research.google.com](https://colab.research.google.com)

**2.** Upload your project files and `images.zip` to the Colab environment.

**3.** Extract the dataset as shown in the [Dataset Instructions](#dataset-instructions) above.

**4.** Run the CNN fine-tuning script:
```bash
!python cnn_finetune.py
```

The script will train or validate the model using the FER2013 dataset and save the resulting weights to `model_3.pth`.

---

## 📊 Results

| Metric | Result |
|---|---|
| Emotion recognition accuracy | ~95% (validation) |
| Inference speed on Raspberry Pi | ~45 ms per frame |
| User satisfaction (emotional appropriateness) | **95%** across 50 test sessions |
| Build cost | ~**£150** |
| Comparable commercial robots | £8,000–£15,000 |

The robot successfully demonstrated real-time emotion detection, empathetic voice replies with tone adaptation, face tracking, and the novel mismatch detection feature across all test sessions.

---

## 🔭 Future Work

- **Mobility** — 4-wheel chassis with SLAM using trajectory planning
- **Expressiveness** — 3-DOF neck and arms for richer gestural communication
- **On-device AI** — Coral TPU integration and battery power for full portability
- **Animated face** — LCD display for dynamic eyes and mouth expressions
- **Real-world trials** — Deployments in care homes and autism support settings

---

## 📄 License

This project was developed as part of a BEng final year project at the University of Hertfordshire. Please contact the author before using any part of this work in commercial or published applications.

---

## 🙏 Acknowledgements

- [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) — facial emotion training data
- [OpenAI Whisper](https://github.com/openai/whisper) — speech-to-text
- [OpenAI GPT-4o](https://platform.openai.com/docs) — conversational AI
- University of Hertfordshire — supervision and facilities
