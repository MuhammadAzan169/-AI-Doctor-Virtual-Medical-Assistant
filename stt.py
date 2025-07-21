import whisper
import sounddevice as sd
import numpy as np
from scipy.signal import resample
from silero_vad import load_silero_vad, get_speech_timestamps
from tts import text_to_speech


def record_until_silence(sample_rate=16000, timeout=10):
    text_to_speech("You can speak now...")
    audio = sd.rec(int(timeout * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("Recording complete.")
    return audio.flatten()


def extract_all_voice_with_padding(audio_int16, sample_rate=16000, padding_ms=200):
    vad_model = load_silero_vad()
    wav = audio_int16.astype(np.float32) / 32768.0

    speech_ts = get_speech_timestamps(wav, vad_model, sampling_rate=sample_rate)
    if not speech_ts:
        return np.array([], dtype=np.float32)

    padding = int(sample_rate * (padding_ms / 1000))
    voiced_segments = []

    for segment in speech_ts:
        start = max(0, segment['start'] - padding)
        end = min(len(wav), segment['end'] + padding)
        voiced_segments.append(wav[start:end])

    return np.concatenate(voiced_segments)


def speech_to_text(model):
    raw_audio = record_until_silence()

    voice_audio = extract_all_voice_with_padding(raw_audio)

    if voice_audio.size == 0:
        return "[No speech detected]"

    audio_16k = resample(voice_audio, int(len(voice_audio) * 16000 / 16000)).astype(np.float32)

    result = model.transcribe(audio_16k, fp16=False, language='en')
    return result["text"]