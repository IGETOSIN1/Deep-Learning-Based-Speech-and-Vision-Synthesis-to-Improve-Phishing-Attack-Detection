import gtts
from playsound import playsound
import random
import string
from PIL import Image 
from pytesseract import pytesseract
import pyaudio
import moviepy.editor as mp
import speech_recognition as sr
from pydub import AudioSegment


def convert_video_to_speech():
    file_path=r"deep_fool_video_files_path"
    video=mp.VideoFileClip(file_path)
    
    audio_file = video.audio 
    audio_file.write_audiofile("audio_path")
    # Initialize recognizer 
    r = sr.Recognizer() 
  
    # Load the audio file 
    with sr.AudioFile("audio_path") as source: 
        data = r.record(source)
    text = r.recognize_google(data) 
  
    return text

def convert_text_to_Audio(text):
    tts = gtts.gTTS(text)
    x = random.choice(string.ascii_letters)
    saved_folder="C:/A-DATA-SETS/mp3s/"
    xsaved_folder=saved_folder+x+".mp3"
    tts.save(xsaved_folder)
    return 
    
def read_image_to_text(pth):
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image_path = pth        
    img = Image.open(image_path) 
    pytesseract.tesseract_cmd = path_to_tesseract 
    text = pytesseract.image_to_string(img) 
    print(text[:-1])

read_image_to_text(r"texs_in_images_format")
