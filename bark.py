from transformers import AutoProcessor, BarkModel
import scipy


processor = AutoProcessor.from_pretrained('suno/bark')
model = BarkModel.from_pretrained('suno/bark')
voice_preset = 'v2/en_speaker_6'

inputs = processor('Hey, welcome to my channel [laughs]', voice_preset=voice_preset)
audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy.squeeze()


sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write(r'outputs/sample_voice.wav', rate=sample_rate, data=audio_array)