import cv2
import time
import threading
import queue
import numpy as np
from openai import OpenAI
import pyttsx3
import base64
from gpiozero import Device, Servo
from gpiozero.pins.lgpio import LGPIOFactory
from collections import deque, Counter
from io import BytesIO
import wave
import sounddevice as sd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

# ───────────── MEMORY PERSISTENCE ─────────────
MEMORY_FILE = "/home/ravi2/Desktop/pico/conversation_memory.json"
conversation_memory = deque(maxlen=20)

# Load previous memory if exists
if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "r") as f:
            conversation_memory.extend(json.load(f))
            print(f"Loaded {len(conversation_memory)} previous messages.")
    except Exception as e:
        print("Error loading memory:", e)

# ───────────── GPIO / SERVOS ─────────────
Device.pin_factory = LGPIOFactory(chip=0)
LEFT_PIN, RIGHT_PIN, WAVE_PIN = 18, 19, 15
left_servo = Servo(LEFT_PIN, min_pulse_width=0.0005, max_pulse_width=0.0025)
right_servo = Servo(RIGHT_PIN, min_pulse_width=0.0005, max_pulse_width=0.0025)
wave_servo = Servo(WAVE_PIN, min_pulse_width=0.0005, max_pulse_width=0.0025)

def set_servos(l, r):
    left_servo.value = l/90
    right_servo.value = -r/90

def go_left(): set_servos(-10,10)
def go_right(): set_servos(10,-10)
def go_center(): set_servos(0,0)

def wave_hand():
    for _ in range(2):
        wave_servo.value = -0.5
        time.sleep(0.3)
        wave_servo.value = 0.5 
        time.sleep(0.3)
    wave_servo.value = 0

# ───────────── OPENAI ─────────────
client = OpenAI(api_key="sk-proj-IDTnAXNKsvR_rATgBQVAZRVSK7yhbl0tvYeFRvgYLZOeNNRBX1llrQyc")

# ───────────── TTS ─────────────
tts_queue = queue.Queue()
engine = pyttsx3.init()
# ───────────── NEW: EMOTION-BASED VOICE TONE ─────────────
def set_voice_by_emotion(emotion):
    if emotion == "Sad":
        engine.setProperty('rate', 120)   # slower, calm
    elif emotion == "Happy":
        engine.setProperty('rate', 170)   # energetic
    elif emotion == "Angry":
        engine.setProperty('rate', 140)
    else:
        engine.setProperty('rate', 150)   # normal


def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None: break
        engine.say(text)
        engine.runAndWait()

threading.Thread(target=tts_worker, daemon=True).start()

def speak_text(t): tts_queue.put(t)

# ───────────── EMOTION MODEL ─────────────
class EmotionDetector(nn.Module):
    def __init__(self, numClasses=7):
        super(EmotionDetector, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.7)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 96, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.7)
        self.fc1 = nn.Linear(96*6*6, 512)
        self.drop3 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(512, numClasses)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.drop1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = self.pool3(x)
        x = self.drop2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x

emotion_labels = ['Happy','Neutral','Sad','Suprise','Angry','Fear','Disgust']   
classifier = EmotionDetector()
classifier.load_state_dict(torch.load("/home/ravi2/Desktop/pico/model_3.pth", map_location='cpu'))
classifier.eval()

# ───────────── AUDIO RECORDING ─────────────
samplerate = 16000
def record_audio(duration=5):
    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    audio = audio.flatten()
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio

def is_silent(audio, threshold=0.01):
    return np.max(np.abs(audio)) < threshold

def is_filler_text(text):
    ignore_phrases = [
        "thank you", "thanks for watching", "like and subscribe",
        "hello", "hi", "okay", "hmm"
    ]
    text = text.lower()
    return any(p in text for p in ignore_phrases)




def audio_to_wav_buffer(audio):
    buf = BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(np.int16(audio * 32767).tobytes())
    buf.seek(0)
    return buf

# ───────────── IMAGE → BASE64 ─────────────
def frame_to_base64(frame, w=320,h=240):
    small = cv2.resize(frame,(w,h))
    _, buffer = cv2.imencode(".jpg", small)
    return base64.b64encode(buffer).decode()

# ───────────── PRODUCT ANALYSIS ───────────
last_product_time = 0
def analyze_product(frame):
    global last_product_time
    if time.time()-last_product_time<5: return
    last_product_time=time.time()
    image_b64 = frame_to_base64(frame)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":[
                {"type":"text","text":"What is this object? Brief description."},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{image_b64}"}}
            ]}],
            max_tokens=50
        )
        desc = response.choices[0].message.content.strip()
        speak_text(f"I see: {desc}")
        print(f"Detected: {desc}")
    except Exception as e:
        print(f"Vision error: {e}")

# ───────────── EMOTION SMOOTHING ─────────────
EMOTION_WINDOW = 10
emotion_buffer = deque(maxlen=EMOTION_WINDOW)
current_emotion = "Neutral"

def smooth_emotion(new_emotion):
    emotion_buffer.append(new_emotion)
    return Counter(emotion_buffer).most_common(1)[0][0]

# ───────────── NEW: EMOTION + VOICE MISMATCH DETECTION ─────────────
def detect_emotion_voice_mismatch(face_emotion, text):
    # Simple keyword-based emotional text analysis
    sad_words = ["sad", "tired", "lonely", "upset", "bad", "depressed"]
    happy_words = ["happy", "good", "great", "excited"]

    text_lower = text.lower()

    if face_emotion == "Happy" and any(w in text_lower for w in sad_words):
        return True, "You look happy, but your words sound sad."
    if face_emotion == "Sad" and any(w in text_lower for w in happy_words):
        return True, "You look sad, but you sound positive."

    return False, ""


# ───────────── UPDATED: EMOTION-AWARE GPT PROMPT ─────────────
def ask_gpt(question, emotion):
    # NEW: system message for emotional intelligence
    system_prompt = {
        "role": "system",
        "content": f"You are a caring social robot. The user currently feels {emotion}. Respond empathetically and briefly."
    }

    # Store user message
    conversation_memory.append({
        "role": "user",
        "content": f"[Emotion: {emotion}] {question}"
    })

    # Build message list
    messages = [system_prompt] + list(conversation_memory)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=12
    )

    reply = response.choices[0].message.content.strip()

    # Store assistant reply
    conversation_memory.append({"role": "assistant", "content": reply})

    return reply


# ───────────── CAMERA SETUP ─────────────
face_cascade = cv2.CascadeClassifier("/home/ravi2/Desktop/pico/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
FRAME_W, FRAME_H = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
frame_center_x, frame_center_y = FRAME_W//2, FRAME_H//2
THRESHOLD = 45
SERVO_DELAY = 0.3
last_servo_move = 0
MOVE_TIMEOUT = 2.0
last_emotion_action = 0
EMOTION_ACTION_DELAY = 5
last_emotion_spoken = None


# MAIN LOOP
try:
    speak_text("Hello! Hasi robot is online.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        command = "C"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.1,5)
        for (x,y,w,h) in faces:
            cx = x + w//2
            offset = cx - frame_center_x
            if offset < -THRESHOLD: command="L"
            elif offset > THRESHOLD: command="R"
            else: command="C"
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(frame,(cx,y+h//2),5,(0,0,255),-1)

            roi = cv2.resize(gray[y:y+h, x:x+w], (48,48))
            roi = torch.tensor(roi/255.0).float().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                pred = classifier(roi)

                probs = torch.softmax(pred, dim=1)        # convert to probabilities
                confidence, idx = torch.max(probs, 1)     # highest confidence emotion
                raw_emotion = emotion_labels[idx.item()]

                if confidence.item() > 0.6:               # only trust strong emotions
                    current_emotion = smooth_emotion(raw_emotion)
                else:
                    current_emotion = "Neutral"

            cv2.putText(frame, current_emotion, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        now = time.time()
        if now - last_servo_move > SERVO_DELAY:
            if command=="L":
                go_left()
                last_servo_move = now
            elif command=="R":
                go_right()
                last_servo_move = now
            else:
                go_center()
        if now - last_emotion_action > EMOTION_ACTION_DELAY:
            if current_emotion == "Happy":
                wave_hand()
                last_emotion_action = now
                last_emotion_spoken = "Happy"

            elif current_emotion == "Sad" and last_emotion_spoken != "Sad":
                speak_text("I hope you feel better soon.")
                go_center()
                last_emotion_action = now
                last_emotion_spoken = "Sad"

            elif current_emotion == "Angry" and last_emotion_spoken != "Angry":
                go_center()
                last_emotion_action = now
                last_emotion_spoken = "Angry"
    

        cv2.circle(frame,(frame_center_x,frame_center_y),5,(255,0,0),-1)
        cv2.putText(frame,f"Command: {command}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow("Hasi Emotion Robot", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'): analyze_product(frame)
        elif key == ord('v'):
            speak_text("Conversation mode activated. Say 'stop' to finish.")
            # conversation mode code continues
            time.sleep(2)
            while True:
                print("Listening....(press B to stopq)")
                
                if cv2.waitKey(1) & 0xFF == ord('b'):
                    speak_text("Conversation stopped.")
                    break

                audio = record_audio(duration=8)

                # 1) Silence check
                if is_silent(audio):
                    print("Silence detected, waiting...")
                    continue   # do not send to Whisper

                wav = audio_to_wav_buffer(audio)

                try:
                    print("Transcribing...")
                    trans = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=("audio.wav", wav)
                    )
                    question = trans.text.strip().lower()
                    print(f"You said: {question}")

                     # 2) Stop word
                    if "stop" in question:
                        speak_text("Conversation stopped. Press V again to talk.")
                        break
                    # Ignore short or filler speech
                    if len(question.split()) < 3 or is_filler_text(question):
                        print("Ignored filler / short sentence.")
                        continue

                    # 3) Emotion mismatch check
                    mismatch, msg = detect_emotion_voice_mismatch(current_emotion, question)
                    if mismatch:
                        speak_text(msg)
                        time.sleep(1)

                        # 4) GPT reply
                    reply = ask_gpt(question, current_emotion)

                         # 5) Emotion-based voice
                    set_voice_by_emotion(current_emotion)
                    speak_text(reply)
                    time.sleep(2)
                    # IMPORTANT: Wait until speaking finishes
                    while engine.isBusy():
                        time.sleep(0.1)

                    time.sleep(0.5)

                except Exception as e:
                    print("Voice error:", e)
                    speak_text("I had trouble hearing you. Please repeat.")
                    # IMPORTANT: Wait until speaking finishes
                    while engine.isBusy():
                        time.sleep(0.1)

            


finally:
    # ───────────── SAFE SHUTDOWN ─────────────
    go_center()
    time.sleep(0.7)
    cap.release()
    cv2.destroyAllWindows()
    tts_queue.put(None)

    # Save conversation memory
    try:
        with open(MEMORY_FILE,"w") as f:
            json.dump(list(conversation_memory), f)
            print("Conversation memory saved.")
    except Exception as e:
        print("Error saving memory:", e)

    print("Hasi robot shutdown complete.")
