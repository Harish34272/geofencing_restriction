import wave, struct, math

def generate_beep(filename="beep.wav", frequency=1000, duration=0.3, volume=0.5, sample_rate=44100):
    n_samples = int(sample_rate * duration)
    with wave.open(filename, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        for i in range(n_samples):
            value = int(volume * 32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            f.writeframes(struct.pack("<h", value))

generate_beep()  # run once to create beep.wav