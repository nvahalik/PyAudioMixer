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
