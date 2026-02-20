import cv2 as cv
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import speech_recognition as sr
from faster_whisper import WhisperModel
import os
import queue
import pickle
from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_groq import ChatGroq

#  trying to understand the archtiecture a bit , i need to use threads to be able to input multiple inputs(audio,visual)  one thread for the brain
#  for the brain which will be created using langgraph ( dont understand how to make it agentic or should use a complex workflow for brian )
# then one thread always for the output and the output needs to be controlled cant overwhelm the patent at any given moment


app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
running=True
visual_input=queue.Queue(maxsize=1)
audio_input=queue.Queue(maxsize=3)
visual_output=queue.Queue(maxsize=1)

#function for my thread 1 , that will constantly take visual input
def input_visual():
    cap=cv.VideoCapture(0)
    while running:
        ret,frame=cap.read()
        if  not ret :
            break
        if visual_input.full():
            try:
                visual_input.get_nowait()
            except queue.Empty:
                pass
        visual_input.put(frame)
    cap.release()

#function for my thread 2 ,take will constantly take audio input
def input_audio():
    r=sr.Recognizer()
    mic = sr.Microphone()
    while running:
        with mic as source:
            r.adjust_for_ambient_noise(source,duration=0.5)
            audio=r.listen(source,phrase_time_limit=15)
            with open("audio.wav",'rb') as f:
                f.write(audio.get_wav_data())
            if audio_input.full():
                try:
                    visual_input.get_nowait()
                except queue.Empty:
                    pass
            visual_input.put(audio)

# now this is where im pondering a bit cause this function i dont knwo where it should be part of my brain or it will intiaalize the langgraph 
def process_visual():
    # name=""
    while running:
        try:
            frame=visual_input.get(timeout=0.6)
        except queue.Empty:
            continue

        faces = app.get(frame)
        # embedding = faces[0].normed_embedding
        # loaded=pickle.load(open('temp_faces.pkl','rb'))
        # for i in loaded:
        #     to_match=i['embeddings']
        #     if np.dot(embedding,to_match)>0.50:
        #         name=i['name']
        


        if visual_output.full():
            try:
                visual_output.get_nowait()
            except queue.Empty:
                pass
        
        visual_output.put((frame,faces))
            
## lets see i think the brain will start from it here 

def recognize():
    name=""
    try:
        frame,faces=visual_output.get(timeout=0.6)
    except queue.Empty:
        return ""
    embedding = faces[0].normed_embedding
    loaded=pickle.load(open('temp_faces.pkl','rb'))
    for i in loaded:
        to_match=i['embeddings']
        if np.dot(embedding,to_match)>0.50:
            name=i['name']
            return name
    
    return "Unknown"


    



