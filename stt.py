from fastapi import FastAPI, File, UploadFile
import speech_recognition as sr
import io
import uvicorn
from pydub import AudioSegment
from text_extract import extract_text

app = FastAPI()

messages = [{
    "role": "system",
    "content": "You are a medical assistant bot. Provide brief, concise answers ONLY to medical questions. "
               "If a user asks a non-medical question, respond with 'Please ask a medical question.' "
               "Keep responses under 100 words."
}]

@app.get("/")
def home():
    return {"message": "Medical Chatbot API is Running!"}

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    recognizer = sr.Recognizer()

    try:
        # Read the uploaded M4A audio file
        audio_data = await file.read()
        audio_file = io.BytesIO(audio_data)

        # Convert M4A to WAV
        audio = AudioSegment.from_file(audio_file, format="m4a")  # Explicitly set M4A format
        audio = audio.set_frame_rate(16000).set_channels(1)  # Standard for speech recognition
        
        # Save to in-memory WAV file
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Recognize speech from the converted WAV file
        with sr.AudioFile(wav_io) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)

        print(text)
        
        res = extract_text(messages,text)
        print(res)
        return res

    except sr.UnknownValueError:
        return {"response": "Could not understand audio"}
    except sr.RequestError as e:
        return {"response": f"Speech recognition service error: {e}"}
    except Exception as e:
        return {"response": f"An error occurred: {e}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
