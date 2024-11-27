import os
import librosa as lr
import numpy as np
import re
from glob import glob
from keras.layers import Dense, LSTM, Activation
from keras.models import Sequential
from keras.optimizers import Adam

# Define constants
SR = 16000  # Sampling Rate
LENGTH = 16  # Amount of blocks for 1 walkthrough
OVERLAP = 8  # Step between samples in amount of blocks
FFT = 1024  # Length of block (64ms)

def filter_audio(audio, sr=16000, energy_threshold=0.35, smooth_window=9):
    """Filter out silence and low-energy segments from audio."""
    
    # Compute Short-Time Fourier Transform (STFT) and convert to decibels
    stft_result = lr.stft(audio, n_fft=2048)
    apower = lr.amplitude_to_db(np.abs(stft_result), ref=np.max)
    
    # Sum energy over frequency bins and normalize
    apsums = np.sum(apower, axis=0)
    apsums -= np.min(apsums)
    apsums /= np.max(apsums)
    
    # Apply a smoothing filter to the energy sums
    smoothed_energy = np.convolve(apsums, np.ones(smooth_window) / smooth_window, mode='same')
    
    # Threshold the energy to distinguish between voice and noise
    energy_mask = smoothed_energy > energy_threshold
    
    # Expand the mask to match the original audio length
    extended_mask = np.repeat(energy_mask, np.ceil(len(audio) / len(energy_mask)))[:len(audio)]
    
    # Apply the mask to filter out noise
    filtered_audio = audio[extended_mask]
    
    return filtered_audio

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

def create_model(list_of_voices, path_of_mod, num_of_epoch=15):
    """Prepare raw data of input list, create, train, and save the model"""

    # Unite all training samples
    X, Y = prepare_audio(*list_of_voices[0])
    for voice in list_of_voices[1:]:
        dx, dy = prepare_audio(*voice)
        X = np.concatenate((X, dx), axis=0)
        Y = np.concatenate((Y, dy), axis=0)
        del dx, dy

    # Shuffle all blocks randomly
    perm = np.random.permutation(len(X))
    X = X[perm]
    Y = Y[perm]

    # Create model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=X.shape[1:]))
    model.add(LSTM(64))
    model.add(Dense(64))
    model.add(Activation('tanh'))
    model.add(Dense(16))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('hard_sigmoid'))

    # Compile and train model
    model.compile(Adam(lr=0.005), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, Y, epochs=num_of_epoch, batch_size=32, validation_split=0.2)
    print(model.evaluate(X, Y))
    model.save(path_of_mod + 'model.hdf5')

    return None

def find_wavs(directory, pattern='**/*.wav'):
    """Recursively find all files matching the pattern"""
    return glob(os.path.join(directory, pattern), recursive=True)

def wav_reader(directory):
    """Find all wav files in directory and compose it in a list of tuples with 'True' mark at target"""
    wav_list = find_wavs(directory)
    res_list = []

    for wav in wav_list:
        temp_list = [wav]

        if re.match(r'.*hamza.*\.wav$', wav):
            temp_list.append(True)
        else:
            temp_list.append(False)

        res_list.append(tuple(temp_list))

    return res_list

if __name__ == "__main__":
    # Specify your directory containing audio files
    directory = 'audio_to_check/'

    # Read the wav files and prepare them for model creation
    voices = wav_reader(directory)

    # Specify the path where you want to save the model
    model_path = 'model/'

    # Create and train the model
    create_model(voices, model_path)
