from gtts import gTTS
import pygame
import tempfile
import os

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang, slow=False)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_path = fp.name
        tts.save(temp_path)

    pygame.mixer.init()
    pygame.mixer.music.load(temp_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.quit()
    os.remove(temp_path)