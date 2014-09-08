PyAudioMixer
============

Advanced Realtime Software Mixer
--------------------------------

Copyright 2008, Nathan Whitehead 
Released under the LGPL

Portions Copyright 2014, Nick Vahalik (KF5ZQE)
Released under the LGPL v2.1


This module implements a realtime sound mixer suitable for use in
games or other audio applications.  It supports loading sounds in
uncompressed WAV format and also MP3 format.  It can mix several
sounds together during playback.  The volume and position of each
sound can be finely controlled.  Sounds are automatically resampled
and stereo converted for correct playback.  Samples can also be looped
any number of times.  Longer sounds can be streamed from a file to
save memory.  In addition, the mixer supports audio input during
playback (if supported in pyaudio with your sound card).

It has been further extended to support multiple simultaneous mixers
which can be controlled independently, frequency and DTMF generators
as well as multiple Microphone input support.

**This code is a work in progress!**

Interfaces and objects are going to be changing drastically as work 
progresses! **Use at your own risk!**

Patches welcome!

Requirements
------------

PyAudio 0.2.0 (or more recent)
http://people.csail.mit.edu/hubert/pyaudio/

NumPy 1.0 (or more recent)
http://numpy.scipy.org/


Optional for MP3 support:

MPEG Audio Decoder (MAD)
http://www.underbit.com/products/mad/

PyMAD bindings for MAD
http://spacepants.org/src/pymad/



Installation
------------

SWMixer is packaged as Python source using distutils.  To install,
run the following command as root:

python setup.py install

For more information and options about using distutils, read:
http://docs.python.org/inst/inst.html


Documentation
-------------

This README file along with the pydoc documentation in the doc/
directory are the documentation for SWMixer.


How can it possibly work in Python?
-----------------------------------

Realtime mixing of sample data is done entirely in Python using the
high performance of array operations in NumPy.  Converting between
sound formats (e.g. mono->stereo) is done using various NumPy
operations.  Resampling is done using the linear interpolation
function of NumPy.  Simultaneous playback and recording is possibly
using PyAudio.

At time of current writing, the latency and CPU utilization of 
PyAudioMixer is slightly better than Audacity running on my test
machine (a 2013 Retina MacBook Pro).


How do I use it?
----------------

@todo - Still need to rewrite the demos, but it works a lot like
swmixer did:

    import PyAudioMixer as pam
    import time
    mixer = pam.Mixer()
    mixer.start()
    snd = pam.Sound("test1.wave")
    snd.play()
    time.sleep(2)
    
Except now, you can have multiple mixers...

    import PyAudioMixer as pam
    mixer1 = pam.Mixer()
    # Just pass the PyAudio device index...
    mixer2 = pam.Mixer(output_device_index=2)
    mixer3 = pam.Mixer(output_device_index=3)
    # etc, etc

See the pydoc documentation for details on all the functions and
for all the options and default values.


Streaming
---------

Normally sounds are loaded entirely into memory before playback
begins.  For long sounds this might result in too much memory being
wasted.  The solution is to create a StreamingSound object.

The interface for StreamingSounds is almost identical to regular
Sounds, but there are some limitations.  Most importantly, the
streaming sound must already be in the correct format for playing.
The samplerate of the streaming sound must match the output
samplerate.  If the output is stereo then the streaming sound must be
stereo.  If the streaming sound is an MP3 then the output must be
stereo.

Here's a very simple example showing a streaming sound along with a
regular sound.
{{{
import swmixer
import time

swmixer.init(samplerate=44100, chunksize=1024, stereo=True)
swmixer.start()
snd1 = swmixer.StreamingSound("Beat_77.mp3")
snd2 = swmixer.Sound("test2.wav")
snd1.play(volume=0.2)
snd2.play()
time.sleep(10.0) #don't quit before we hear the sound!
}}}

StreamingSounds have most of the functionality of regular Sounds, but
some operations are not allowed.  For example, WAV streams do not
allow arbitrary jumping to a position; MP3 streams do.  MP3 streams
allow checking the total length with get_length(), while WAV streams
do not.  (However WAV Sounds do have get_length()).

You can have any number of StreamingSounds and Sounds playing at once.


Explicit Tick Interface
-----------------------

Instead of calling swmixer.start() you may also call swmixer.tick() every
frame in your main loop.  This gives you greater control over synchronizing
the video framerate with audio events for music applications and games.

The samplerate and chunksize will limit your framerate.  If you set
the samplerate to 44100 samples per second, and each chunk is 1024
samples, then each call to swmixer.tick() will process 1024 samples
corresponding to 0.0232 seconds of audio.  This will lock your
framerate at 1/.0232=43.1 frames per second.  If you call
swmixer.tick() faster than this, that's OK, it will just block until
more audio can be send to the soundcard.  If you call swmixer.tick()
slower than 43.1 times a second, there will be audio glitches.

Note that by choosing your samplerate and chunksize wisely you can get
any framerate you want.  Larger chunksizes correspond to slower
framerates.  You may also call swmixer.tick() every other frame, or
every third frame.  This way your video framerate will be a fixed
multiple of your audio framerate.

Here is a silly example showing a moving green square with a
background sound.  The square should move at 43 pixels / second.

{{{
import sys
import swmixer
import pygame

swmixer.init(samplerate=44100, chunksize=1024, stereo=False)
snd = swmixer.Sound("test1.wav")
pygame.display.init()
screen = pygame.display.set_mode((1024, 768))

snd.play()
x = 0
while True:
      swmixer.tick()
      x += 1
      screen.fill((0, 0, 0))
      pygame.draw.rect(screen, (0, 255, 0), (x, 100, 50, 50))
      pygame.display.flip()
      for evt in pygame.event.get():
          if evt.type == pygame.QUIT: sys.exit()
}}}

You can also call swmixer.set_buffersize(size) at any time to change
the buffer size and thus change the framerate.  Switching the buffer
size to 512 will double the framerate.  SWMixer does not impose any
requirements on the buffer size, it can be anything.  As the buffer
size gets smaller you will have to call swmixer.tick() very quickly
to avoid audio glitches.


Recording
---------

**Currently being overhauled.**


Usage with pygame?
------------------

????


Bugs and Limitations
--------------------

Always outputs in 16-bit mode.

Cannot deal with 24-bit WAV files, but CAN handle 32-bit ones
(limitation of NumPy).

Resampling can be slow for longer files.

Does not detect samplerates that differ from requested samplerates.
I.e.  if you request a rate your card cannot handle, you might get
incorrect playback rates.

Currently there is no way to limit the number of sounds mixed at once
to prevent excessive CPU usage.

No way to pan mono sounds to different positions in stereo output.

StreamingSounds may not be sample accurate for looping and setting
position.

Threading behavior may not be optimal on some platforms.
