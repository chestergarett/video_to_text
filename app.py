import datetime
import whisper
import json

model = whisper.load_model('base.en')
option = whisper.DecodingOptions(language='en',fp16=False)
result = model.transcribe(f'inputs/BPI Hack-a-thon Together, Not A-Loan.mp4')

output_file = r'outputs/transcription_result.json'
with open(output_file, 'w') as json_file:
    json.dump(result, json_file, indent=4)

print(result['text'])