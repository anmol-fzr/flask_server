from flask import Flask, request, jsonify
from spafe.frequencies.dominant_frequencies import get_dominant_frequencies
import librosa
import numpy as np
import io

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

@app.route('/', methods=['POST'])
def calculate_dominant_frequencies_and_decibels():
    audio_buffer = request.data
    audio_file = io.BytesIO(audio_buffer)
    try:
        sig, fs = librosa.load(audio_file, sr=None)

        # Calculate average decibels using RMS amplitude
        rms = librosa.feature.rms(y=sig)
        db = 20 * np.log10(rms / np.iinfo(np.int16).max)  # Reference to full scale
        average_db = np.mean(db)  # Calculate average decibel value

        dominant_frequencies = get_dominant_frequencies(sig, fs,
                                                       butter_filter=False,
                                                       lower_cutoff=0,
                                                       upper_cutoff=fs/2,
                                                       nfft=512,
                                                       win_len=0.020,
                                                       win_hop=0.010,
                                                       win_type="hamming")

        dominant_frequencies_list = list(dominant_frequencies)

        return jsonify({'data': dominant_frequencies_list, 'db': average_db.tolist()})
    except Exception as error:
        print(error)
        return jsonify({'name': 'flask'})


if __name__ == '__main__':
    app.run()
