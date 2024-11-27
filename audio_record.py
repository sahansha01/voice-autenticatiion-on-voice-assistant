import pyaudio
import wave

def voice_recorder(output_filename, seconds_of_audio):
    """Record audio and save it as a WAV file."""
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2  # Stereo
    fs = 16000  # Sample rate

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    input_device_index=None,  # Specify device index if needed
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for specified duration
    for _ in range(0, int(fs / chunk * seconds_of_audio)):
        try:
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
        except IOError as e:
            print(f"Error occurred while recording: {e}")
            break

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))

if __name__ == "__main__":
    for i in range(34,44):
        output_filename = f"audio_to_check/hamza{i}.wav"  # Output WAV file name
        duration = 5  # Duration of recording in seconds
        voice_recorder(output_filename, duration)
