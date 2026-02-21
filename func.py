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
from groq import RateLimitError
from neo4j import GraphDatabase
from dotenv import load_dotenv
import threading
import pyttsx3
import io
from pydantic import BaseModel,Field

load_dotenv(override=True)
keys=os.getenv("groq_key")
groq_key=[ i.strip()  for i in keys.split(",") ]
URI=""
AUTH=""
#  trying to understand the archtiecture a bit , i need to use threads to be able to input multiple inputs(audio,visual)  one thread for the brain
#  for the brain which will be created using langgraph ( dont understand how to make it agentic or should use a complex workflow for brian )
# then one thread always for the output and the output needs to be controlled cant overwhelm the patent at any given moment


app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
engine = pyttsx3.init('sapi5')
engine.setProperty('rate', 160) 
engine.setProperty('volume', 1.0)
voices = engine.getProperty('voices')
if len(voices) > 1:
    engine.setProperty('voice', voices[1].id)
running=True
visual_input=queue.Queue(maxsize=1)
audio_input=queue.Queue(maxsize=3)
visual_output=queue.Queue(maxsize=1)
audio_output=queue.Queue(maxsize=3)
current_key=0
llm=ChatGroq(model="llama-3.3-70b-versatile",api_key=groq_key[current_key])
patient_name="Johnny"
transcription=""





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
            # with open("audio.wav",'wb') as f:
            #     f.write(audio.get_wav_data())
            raw_wav_bytes = audio.get_wav_data()
            if audio_input.full():
                try:
                    audio_input.get_nowait()
                except queue.Empty:
                    pass
            audio_input.put(raw_wav_bytes)

# now this is where im pondering a bit cause this function i dont knwo where it should be part of my brain or it will intiaalize the langgraph 
def process_visual():
    while running:
        try:
            frame=visual_input.get(timeout=0.6)
        except queue.Empty:
            continue

        faces = app.get(frame)
        
        if visual_output.full():
            try:
                visual_output.get_nowait()
            except queue.Empty:
                pass
        
        visual_output.put((frame,faces))

def process_audio():
    global transcription
    while running:
        while not audio_input.empty():
            audio_item = audio_input.get()
            audio_file_like = io.BytesIO(audio_item)
            segments, _ = model.transcribe(audio_file_like, beam_size=5)
            for segment in segments:
                transcription += segment.text + " "
            

class RelationsGraph(BaseModel):
   last_convo: str = Field(
        description='''
            Summarize the transcript in about  1-2 line 
            '''

    )
   relations: str=Field(
       description='''
            Study the transcript and determine the relation between the patient and person , if none is identified return unknown
'''
   )
## lets see i think the brain will start from it here 
class Alzheimer(TypedDict):
    name:str
    relation:str
    last_convo:str
    #live_convo:str

def recognize(state:Alzheimer):
    name=""
    global embedding
    try:
        frame,faces=visual_output.get(timeout=0.6)
    except queue.Empty:
        return {'name':""}
    if len(faces)==0:
        return {'name':""}
    embedding = faces[0].normed_embedding
    loaded=pickle.load(open('temp_faces.pkl','rb'))
    for i in loaded:
        to_match=i['embeddings']
        if np.dot(embedding,to_match)>0.50:
            name=i['name']
            return {'name':name}
    return {'name':"Unknown"}

## atill have to work on handling llm calling and return factor in this function
def identification(state:Alzheimer):
    try:
        res=llm.invoke(f'''Identify the name of the person from the following conversation if name not found give a random 12 digitnumber and only number **DO NOT RETURN THE PATIENT NAME{patient_name}**  
                   **RETURN ONLY THE NAME , NO EXTRA INFORMATION**
                   conversation:
                   {transcription}
                   ''').content
    except RateLimitError:
        current_key+=1
        return{'name':"failed"}
    dr={'name':res,'embeddings':embedding}
    with open('temp_faces.pkl', 'ab') as file:
        pickle.dump(dr, file)
    return {'name':res}

# should i make it a tol for the llm to use
def getraginfo(state:Alzheimer):
    name=state["name"]
    relation=""
    last_convo=""
    live_convo=""
    try:# will check if person is already in rag or not
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            records,summary,key=driver.execute_query("MATCH (n:Person[{name:$name1}])-[r:KNOWS]->(m:Person{name:$name2}) RETURN n,r,m",name1=patient_name,name2=name,database_="neo4j")
        for record in records:
            x=record.data()
            source=x.get("m")
            relation=source["relation"]
            last_convo=source["last_convo"]
        return{'relation':relation,'last_convo':last_convo}
    except Exception as e:
        pass
    return {'relation':'','last_convo':''}
#this is gonna be the  most important function so the quality of this function should be aas good as it can be
def live_help(state:Alzheimer):
    status_check_prompt=f'''You are an expert medical assistant for alzheimer patients , 
    look at the conversation and return yes or no if you think the patient needs help of not {transcription}'''
    try:
        res=llm.invoke(status_check_prompt).content
        if "no" in res.lower():
            transcription=transcription[int(len(transcription)/2):]
            return
        else:
            name = state.get("name", "")
            relation = state.get("relation", "")
            last_convo = state.get("last_convo", "")
            system_prompt = f"""You are an invisible cognitive assistant for an Alzheimer's patient named {patient_name}.
                            Your job is to provide short, comforting whispers in their ear to help them navigate social situations.
                            
                            CURRENT CONTEXT:
                            - Person in front of them: {name if name and name != 'Unknown' else 'An unrecognized person'}
                            - Relation to patient: {relation if relation else 'Unknown'}
                            - Memory of last conversation: {last_convo if last_convo else 'No previous memory context available.'}
                            - What the person just said: "{transcription}"
                            
                            INSTRUCTIONS:
                            1. KEEP IT EXTREMELY BRIEF (Max 1-2 short sentences).
                            2. Do not overwhelm {patient_name}.
                            3. Gently remind {patient_name} who they are talking to and suggest a simple reply.
                            4. Speak directly to {patient_name} (e.g., "Johnny, this is your son Vihaan...")"""
            res = llm.invoke(system_prompt).content
            if audio_output.full():
                try:
                    audio_output.get_nowait()
                except queue.Empty:
                    pass
            audio_output.put(res)
    except RateLimitError:
        current_key+=1
    return

def creategraphinfo(state:Alzheimer):
    name=state["name"]
    last_convo=""
    relation=""
    try:
        structured_llm=llm.with_structured_output(RelationsGraph)
        res=structured_llm.invoke(f"You are an expert relation analyzer , analyze the relation from the transcript and before that summarize the convo Transcript:{transcription}")
        last_convo=res.last_convo
        relation=res.relations
    except Exception as e:
        pass
    query='''
Create
'''



def output_audio():
    while running:
        try:
            text=audio_output.get(timeout=0.6)
        except queue.Empty:
            continue
        engine.say(text)
        engine.runAndWait()

def condition_check_recognize(state:Alzheimer):
    name=state['name']
    if name =="":
        return recognize
    if name== "Unknown":
        return identification

    



