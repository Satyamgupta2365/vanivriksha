import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
from pydub import AudioSegment

def load_audio(file_path):
    audio, _ = librosa.load(file_path, sr=16000)
    return audio

# Convert mp3 to wav
def convert_to_wav(mp3_path):
    audio = AudioSegment.from_mp3(mp3_path)
    wav_path = mp3_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    return wav_path

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")
qa_pipeline = pipeline("question-answering")

# Process and transcribe audio files
audio_files = [
    r"C:\Users\shubh\OneDrive\Desktop\aiml\dataset\SandalWoodNewsStories_1.mp3",
    r"C:\Users\shubh\OneDrive\Desktop\aiml\dataset\SandalWoodNewsStories_2.mp3",
    r"C:\Users\shubh\OneDrive\Desktop\aiml\dataset\SandalWoodNewsStories_6.mp3"
]

audio_transcriptions = {}
for file in audio_files:
    if file.endswith(".mp3"):
        file = convert_to_wav(file)
    audio = load_audio(file)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    audio_transcriptions[file] = transcription
    print(f"Transcribed {file}: {transcription}")

# Question-Answering from transcriptions
def get_answer(question):
    best_answer = None
    best_score = 0
    for file, transcription in audio_transcriptions.items():
        result = qa_pipeline(question=question, context=transcription)
        if result['score'] > best_score:
            best_answer = result['answer']
            best_score = result['score']
    return best_answer

# Example usage of the QA system
question = "How to cultivate sandalwood?"
answer = get_answer(question)
print("Final Answer:", answer)
