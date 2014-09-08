import PyAudioMixer as pam 
import time

mixer = pam.Mixer(stereo=False)
mixer.start()

snd = pam.Sound(mixer, "test2.wav")
chan = snd.play(fadein=22000.0) #fade in sound over 0.5 seconds
time.sleep(1.0)
# rewind 20000 samples now just for kicks
chan.set_position(chan.get_position() - 20000)
time.sleep(1.0)
chan.stop()
time.sleep(1.0)
