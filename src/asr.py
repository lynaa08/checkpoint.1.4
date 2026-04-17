import whisper

model = whisper.load_model("base")
result = model.transcribe(r"C:\Users\tassili\Documents\checkpoint.1.4\data\audio.wav")

print(result["text"])