# Import library:
import speech_recognition as sr
from gtts import gTTS
import os

# Add Speech recognition:
def audio_reader():
    reader = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak...")
        audio = reader.listen(source)
    try:
        return reader.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I don't understand"
    except sr.RequestError:
        return "Could not process"
    
# Output with Audio Response:
def speak_output(text, filename="audiResponse.mp3"):
    tts = gTTS(text=text)
    tts.save(filename)
    os.system(f"start{filename}")
    