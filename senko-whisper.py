import senko 
import whisper
import json

#----------CONFIG-----#

AUDIO_FILE = "call_mono.wav"
WHISPER_MODEL = "base"
DEVICE = "auto"

#---------initialize senko model-----#

print("Loading Senko model...")
diaizer = senko.Diarizer(device=DEVICE, warmup=True, quiet=False)

#--------run diarization------#

print(f"Diarizing {AUDIO_FILE}...")
dia_result = diaizer.diarize(AUDIO_FILE, generate_colors=False)

segments = dia_result["merged_segments"] 

#----------initialize whisper model-----#

model = whisper.load_model(WHISPER_MODEL)

#--------run transcription------#

print(f"Transcribing {AUDIO_FILE}...")
whisper_result = model.transcribe(AUDIO_FILE)

#--------combine diarization and transcription------#

diarized_transcript = []

for seg in whisper_result["segments"]:
    mid_time = (seg["start"] + seg["end"]) / 2

    #find speaker
    
    speaker = "Unknown"
    for s in segments:
        if s["start"] <= mid_time <= s["end"]:
            speaker_label = s["speaker"]
            break 

    diarized_transcript.append({
        "start": seg["start"],
        "end": seg["end"],
        "speaker": speaker_label,
        "text": seg["text"].strip()
    })  

#--------output diarized transcript------#

print("\n ----DIARIZED TRANSCRIPT---- \n")
for entry in diarized_transcript:
    print(f"[{entry['start']:.2f} - {entry['end']:.2f}] {entry['speaker']}: {entry['text']}")


#----------json output-----#

with open("diarized_transcript.json", "w") as f:
    json.dump(diarized_transcript, f, indent=2)

print("\nDiarized transcript saved to diarized_transcript.json")    