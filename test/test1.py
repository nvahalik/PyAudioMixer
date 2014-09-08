import PyAudioMixer as pam
import time

mixer = pam.Mixer(samplerate=44100, chunksize=1024, stereo=False)
mixer.start()

snd1 = pam.Sound(mixer, "test1.wav")
snd2 = pam.Sound(mixer, "test2.wav")
snd1.play(loops=-1)
snd2.play()
time.sleep(10.0) #don't quit before we hear the sound!
