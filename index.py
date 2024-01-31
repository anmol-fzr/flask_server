from flask import Flask, request, jsonify
from spafe.frequencies.dominant_frequencies import get_dominant_frequencies
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['POST'])
def calculate_dominant_frequencies():
    audio_file = request.files['audio']
 
    # Read the audio file using a suitable library (e.g., librosa, soundfile, wave)
    fs, sig = read_audio_file(audio_file)  # Replace with your preferred audio reading function

    # Compute dominant frequencies using spafe
    dominant_frequencies = get_dominant_frequencies(sig,
                                                    fs,
                                                    butter_filter=False,
                                                    lower_cutoff=0,
                                                    upper_cutoff=fs/2,
                                                    nfft=512,
                                                    win_len=0.020,
                                                    win_hop=0.010,
                                                    win_type="hamming")

    # Convert dominant frequencies to a list for JSON serialization
    dominant_frequencies_list = list(dominant_frequencies)

    return jsonify({'dominant_frequencies': dominant_frequencies_list})

if __name__ == '__main__':
    app.run()
