import sys
import swmixer
import numpy

# Test of multiple soundcards
# You need two soundcards plus a USB mic for this one

swmixer.init(samplerate=44100, chunksize=1024, stereo=False, microphone=True, output_device_index = 1, input_device_index = 2)
snd = swmixer.Sound("test1.wav")
snd.play(loops=-1)

micdata = []
frame = 0

while True:
    swmixer.tick()
    frame += 1
    if frame < 50:
        micdata = numpy.append(micdata, swmixer.get_microphone())
    if frame == 50:
        micsnd = swmixer.Sound(data=micdata)
        micsnd.play()
        micdata = []
        frame = 0

    

