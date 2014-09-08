import swmixer
import time

swmixer.init(samplerate=44100, chunksize=1024, stereo=False)
swmixer.start()
snd1 = swmixer.Sound("test1.wav")
snd2 = swmixer.Sound("test2.wav")
snd1.play(loops=-1)
snd2.play()
time.sleep(10.0) #don't quit before we hear the sound!
