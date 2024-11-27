import os
import librosa as lr
import numpy as np
from keras.models import load_model
from audio_record import *
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

SR = 16000  # Sampling Rate
LENGTH = 16  # Amount of blocks for 1 walkthrough
OVERLAP = 8  # Step between samples in amount of blocks
FFT = 1024  # Length of block (64ms)

def filter_audio(audio):
    """Filter every audio file in raw data in several parameters"""
    
    # Calculate voice energy for every 123 ms block
    apower = lr.amplitude_to_db(np.abs(lr.stft(audio, n_fft=2048)), ref=np.max)

    # Summarize energy of every rate, normalize
    apsums = np.sum(apower, axis=0)
    apsums -= np.min(apsums)
    apsums /= np.max(apsums)

    # Smooth the graph for saving short spaces and pauses, remove sharpness
    apsums = np.convolve(apsums, np.ones((9,)), 'same')
    # Normalize again
    apsums -= np.min(apsums)
    apsums /= np.max(apsums)

    # Set noise limit to 35% over voice
    apsums = np.array(apsums > 0.35, dtype=bool)

    # Extend the blocks every on 125ms
    # before separated samples (2048 at block)
    apsums = np.repeat(apsums, np.ceil(len(audio) / len(apsums)))[:len(audio)]

    return audio[apsums]

def prepare_audio(a_name, target=False):
    """Audio tokenizer for further neural model usage"""
    print('Loading %s' % a_name)
    audio, _ = lr.load(a_name, sr=SR)
    audio = filter_audio(audio)  # Removing silence and spaces between words
    data = lr.stft(audio, n_fft=FFT).swapaxes(0, 1)  # Export spectrogram
    samples = []

    for i in range(0, len(data) - LENGTH, OVERLAP):
        samples.append(np.abs(data[i:i + LENGTH]))  # Create training sample

    results_shape = (len(samples), 1)
    results = np.ones(results_shape) if target else np.zeros(results_shape)

    return np.array(samples), results

def check_access(target_path, mod_path):
    """Check access status of target with pre-trained model"""
    model = load_model(os.path.join(mod_path, 'model.hdf5'))

    new_audio, _ = prepare_audio(target_path)  # Note: No need for 'target' argument here
    
    # Check model input shape
    print("Model input shape:", model.input_shape)
    print("New audio input shape:", new_audio.shape)
    if len(new_audio) == 0:
        print("No valid audio samples found after filtering.")
        return False
    
    # Print sample of new_audio for debugging
    # Predict with the model
    predictions = model.predict(new_audio)
    
   
    # Sum of predictions
    val_sum = np.sum(predictions)
    
    # Print the sum and percentage
   # print(f"Sum of predictions: {val_sum}")
    percentage = 100 * (val_sum / len(predictions))
    print(f"Access percentage: {percentage:.3f}%")

    # Determine access based on the percentage threshold
    if percentage > 85.0:
       # print('Access is allowed')
        return True
    else:
      #  print('Access is denied')
        return False

def main():
    
    target_to_check = 'raw_data_wav/new.wav'
    voice_recorder(target_to_check , 5)
    check_access(target_to_check, 'model/')

if __name__ == "__main__":
    main()
