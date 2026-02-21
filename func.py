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
import time

load_dotenv(override=True)
keys=os.getenv("groq_key")
groq_key=[ i.strip()  for i in keys.split(",") ]
URI=os.getenv("URI")
db=os.getenv("Database")
pwd=os.getenv("pwd")
AUTH=(db,pwd)


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
patient_name="Harry"
transcription=""


try:
    global_known_faces = pickle.load(open('temp_faces.pkl', 'rb'))
except (EOFError, FileNotFoundError):
    global_known_faces = []
current_display_frame = None
current_display_name = "Scanning..."
current_display_box = None


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
        time.sleep(0.3)

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
    global embedding, current_display_name, current_display_box, current_display_frame
    try:
        frame,faces=visual_output.get(timeout=0.6)
        current_display_frame = frame.copy()
    except queue.Empty:
        current_display_name = "Scanning..."
        current_display_box = None
        return {'name':""}
    if len(faces)==0:
        current_display_name = "No faces detected"
        current_display_box = None
        return {'name':""}
    embedding = faces[0].normed_embedding
    current_display_box = faces[0].bbox.astype(int)
    # loaded=pickle.load(open('temp_faces.pkl','rb'))
    #for i in loaded:
    for i in global_known_faces:
        to_match=i['embeddings']
        if np.dot(embedding,to_match)>0.50:
            name=i['name']
            current_display_name = name
            return {'name':name}
    return {'name':"Unknown"}

## atill have to work on handling llm calling and return factor in this function
def identification(state:Alzheimer):
    global current_key, llm
    try:
        res=llm.invoke(f'''Identify the name of the person from the following conversation if name not found give a random 12 digitnumber and only number **DO NOT RETURN THE PATIENT NAME{patient_name}**  
                   **RETURN ONLY THE NAME , NO EXTRA INFORMATION**
                   conversation:
                   {transcription}
                   ''').content
    except RateLimitError:
        
        current_key = (current_key + 1) % len(groq_key) 
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key[current_key])
        return{'name':"failed"}
    dr={'name':res,'embeddings':embedding}
    global_known_faces.append(dr)
    with open('temp_faces.pkl', 'wb') as file:
        pickle.dump(global_known_faces, file)
    return {'name':res}

# should i make it a tol for the llm to use
def getraginfo(state:Alzheimer):
    name=state["name"]
    relation=""
    last_convo=""
    #live_convo=""
    try:# will check if person is already in rag or not
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            records,summary,key=driver.execute_query("MATCH (n:Person{name:$name1})-[r:KNOWS]->(m:Person{name:$name2}) RETURN n,r,m",name1=patient_name,name2=name,database_="neo4j")
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
    global current_key, llm,transcription
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
        
        current_key = (current_key + 1) % len(groq_key) 
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key[current_key])
    return

def creategraphinfo(state:Alzheimer):
    global transcription,current_key, llm
    name=state["name"]
    last_convo=""
    relation=""
    try:
        structured_llm=llm.with_structured_output(RelationsGraph)
        res=structured_llm.invoke(f"You are an expert relation analyzer , analyze the relation from the transcript and before that summarize the convo Transcript:{transcription}")
        last_convo=res.last_convo
        relation=res.relations
    except RateLimitError:
        current_key = (current_key + 1) % len(groq_key) 
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key[current_key])
    query='''
        MERGE (e:Person{name:$name1})

        MERGE (f:Person{name:$name2})
        ON CREATE
        SET f.relation=$relation,
            f.last_convo=$last_convo

        MERGE (e)-[r:KNOWS]->(f)
        '''
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        try:
            driver.execute_query(
                query,
                name1=patient_name,
                name2=name,
                relation=relation,
                last_convo=last_convo,
                database_="neo4j"
            )
        except Exception as e:
            pass



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
        return END
    elif name== "Unknown":
        return "identification"
    else:
        return "getraginfo"

def route_after_identification(state: Alzheimer):
    name = state.get("name", "")
    if name == "failed" or name == "":
        return END 
    return "getraginfo"
def route_after_help(state: Alzheimer):
    global transcription
    if len(transcription) > 300: 
        return "creategraphinfo"
    return END


workflow = StateGraph(Alzheimer)

workflow.add_node("recognize", recognize)
workflow.add_node("identification", identification)
workflow.add_node("getraginfo", getraginfo)
workflow.add_node("live_help", live_help)
workflow.add_node("creategraphinfo", creategraphinfo)

workflow.add_edge(START, "recognize")
workflow.add_conditional_edges("recognize", condition_check_recognize)
workflow.add_conditional_edges("identification", route_after_identification)
workflow.add_edge("creategraphinfo", END)
workflow.add_edge("getraginfo", "live_help")
workflow.add_conditional_edges("live_help", route_after_help)


if __name__ == "__main__":
    
    brain = workflow.compile()

    print("[SYSTEM] Starting sensory threads...")
    threads = [
        threading.Thread(target=input_visual, daemon=True),
        threading.Thread(target=input_audio, daemon=True),
        threading.Thread(target=process_visual, daemon=True),
        threading.Thread(target=process_audio, daemon=True),
        threading.Thread(target=output_audio, daemon=True)
    ]
    for t in threads:
        t.start()

    print("[SYSTEM] Brain online. Starting UI and Main Loop...")
    
    last_known_name = ""

    try:
        while running:
            result = brain.invoke({"name": "", "relation": "", "last_convo": ""})
            
            if result and result.get("name") not in ["", "Unknown", "failed"]:
                last_known_name = result["name"]

            if current_display_frame is not None:
                display_img = current_display_frame.copy()
                
                if current_display_box is not None:
                    x1, y1, x2, y2 = current_display_box
                    cv.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    cv.rectangle(display_img, (x1, y1-35), (x2, y1), (0, 255, 0), cv.FILLED)
                    cv.putText(display_img, current_display_name, (x1+5, y1-10), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

                cv.imshow("Alzheimer's Assistant POV", display_img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break



    except KeyboardInterrupt:
        print("\n[SYSTEM] Manual interruption.")
        
    finally:
        print("[SYSTEM] Shutting down...")
        running = False
        cv.destroyAllWindows()
        if last_known_name != "" and len(transcription) > 50:
            print(f"[SYSTEM] Saving final memory of {last_known_name} to Neo4j...")
            creategraphinfo({"name": last_known_name})
            
        print("[SYSTEM] Goodbye.")