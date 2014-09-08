import sys
import swmixer
import pygame

swmixer.init(samplerate=44100, chunksize=1024, stereo=False)
snd = swmixer.Sound("test1.wav")
pygame.display.init()
screen = pygame.display.set_mode((1024, 768))

snd.play(loops=-1)

# Demonstrate framerate changes
# Starts at 44100 samples per second, each frame is 1024 samples
# so it takes .0232 seconds per frame, or 43.1 frames per second.
# After 200 frames, the buffer is changed to 512 samples.
# This means each frame takes .0116 seconds, or 86.2 frames per second.
# So the moving square doubles its speed after 200 frames.
# The audio should sound normal the whole time.

x = 0
while True:
      swmixer.tick()
      x += 1
      if x == 200:
            swmixer.set_chunksize(512)
      screen.fill((0, 0, 0))
      pygame.draw.rect(screen, (0, 255, 0), (x, 100, 50, 50))
      pygame.display.flip()
      for evt in pygame.event.get():
          if evt.type == pygame.QUIT: sys.exit()
