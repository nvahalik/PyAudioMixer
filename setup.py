from distutils.core import setup, Extension

setup(name = "SWMixer",
      author = "Nathan Whitehead",
      author_email = "nwhitehe@gmail.com",
      version = "0.1.4",
      url = "http://code.google.com/p/pygalaxy/",
      py_modules = ['swmixer'],
      description = "An advanced software mixer for sound playback and recording",
      long_description = '''
This module implements a realtime sound mixer suitable for use in
games or other audio applications.  It supports loading sounds in
uncompressed WAV format and also MP3 format.  It can mix several
sounds together during playback.  The volume and position of each
sound can be finely controlled.  Sounds are automatically resampled
and stereo converted for correct playback.  Samples can also be looped
any number of times.  Longer sounds can be streamed from a file to
save memory.  In addition, the mixer supports audio input during
playback (if supported in pyaudio with your sound card).
''',
      )
