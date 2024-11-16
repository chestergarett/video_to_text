import datetime
import whisper

model = whisper.load_model('base.en')
option = whisper.DecodingOptions(language='en',fp16=False)
result = model.transcribe(f'inputs/BPI Hack-a-thon Together, Not A-Loan.mp4')

print(result['text'])