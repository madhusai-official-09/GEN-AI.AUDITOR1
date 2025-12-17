import whisper 
import webrtcvad 
import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import medfilt
import sklearn

#--------- configurable parameters ---------#
AUDIO_FILE = "audio.wav"
WHISPER_MODEL = "base"
SR = 16000
WINDOW = 0.5 
HOP = 0.25 
MIN_TURN = 0.6
SMOOTH_KERNEL = 3
OVERLAP_ENERGY_THRESHOLD = 0.02

#----------load audio file -----------#

wav = preprocess_wav(AUDIO_FILE)
duration = len(wav) / SR

#----------initialize vad ---------# 
vad = webrtcvad.Vad(2)

#---------- Vad helper----------#
def vad_speech_ratio(audio, sr, vad, frame_ms=30):
    frame_len = int(sr * frame_ms / 1000)
    if len(audio) < frame_len:
        return 0.0

    speech_frames = 0
    total_frames = 0

    for i in range(0, len(audio) - frame_len, frame_len):
        frame = audio[i:i+frame_len]
        pcm = (frame * 32767).astype(np.int16).tobytes()
        try:
            if vad.is_speech(pcm, sr):
                speech_frames += 1
            total_frames += 1
        except:
            continue

    return speech_frames / max(total_frames, 1)

#---------- sliding winow ---------#

windows, times = [], []
t= 0.0 
while t + WINDOW <= duration:
    s = int(t * SR)
    e = int((t + WINDOW) * SR)
    chunk = wav[s:e]

    speech_ratio = vad_speech_ratio(chunk, SR, vad)
    if speech_ratio > 0.3: 
        windows.append(chunk)
        times.append((t, t + WINDOW))

    t += HOP


 #--------- speaker embedding ---------#

encoder = VoiceEncoder()
embeddings = np.array([encoder.embed_utterance(w) for w in windows])

#--------- clustering-------#
sk_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
if sk_version >= (1, 2):
    cluster = AgglomerativeClustering(
        n_clusters=2,
        metric="cosine",
        linkage="average"
    )
else:
    cluster = AgglomerativeClustering(
        n_clusters=2,
        affinity="cosine",
        linkage="average"
    )

labels = cluster.fit_predict(embeddings)


#--------- smoothing -------#
labels = medfilt(labels, kernel_size=SMOOTH_KERNEL)

# ---------- min turn logic-----------#
final_labels = labels.copy()
last_label = labels[0]
last_time = times[0][0]

for i in range(1, len(labels)):
    if labels[i] != last_label:
        if times[i][0] - last_time < MIN_TURN:
            final_labels[i] = last_label
        else:
            last_label = labels[i]
            last_time = times[i][0]


#----------- whisper ------------#
model = whisper.load_model(WHISPER_MODEL) 
result = model.transcribe(AUDIO_FILE) 

#---------- overlapping handler --------# 

speaker_map = {0: "Speaker 1", 1: "Speaker 2"}
print("\n ----DIARIZED TRANSCRIPT---- \n")
for seg in result["segments"]:
    mid = (seg["start"] + seg["end"]) / 2
    idx = min(range(len(times)), key=lambda i: abs((times[i][0] + times[i][1]) / 2 - mid))

    speaker = speaker_map[final_labels[idx]]

    energy = np.mean(windows[idx] ** 2)
    if energy < OVERLAP_ENERGY_THRESHOLD:
        speaker += " (overlap)"

    print(f"[{seg['start']:.2f} - {seg['end']:.2f}] {speaker}: {seg['text'].strip()}")    


