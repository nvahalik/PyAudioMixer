import swmixer
import time

swmixer.init(samplerate=44100, chunksize=1024, stereo=True)
swmixer.start()
snd1 = swmixer.StreamingSound("Beat_77.mp3")
snd2 = swmixer.Sound("test2.wav")
print snd1.get_length(), snd2.get_length()
snd1.play(volume=0.2)
snd2.play()
time.sleep(10.0) #don't quit before we hear the sound!
