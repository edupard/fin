import os, sys
import pygame
from pygame.locals import *

# set SDL to use the dummy NULL video driver,
#   so it doesn't need a windowing system.
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame.transform


if 1:
    # some platforms might need to init the display for some parts of pygame.
    # import pygame.display

    pygame.display.init()
    pygame.display.set_mode((1, 1))


def scaleit(fin, fout, w, h):
    screen = pygame.Surface((400, 400), pygame.SRCALPHA, 32)
    blue = (0, 0, 255)
    red = (255, 0, 0)
    pygame.draw.rect(screen, blue, (0, 0, 10.5, 10.5), 0)
    pygame.draw.line(screen, red, (-10,-10), (100, 100), 1)
    pygame.display.flip()
    pygame.image.save(screen, fout)

    # i = pygame.image.load(fin)
    #
    # if hasattr(pygame.transform, "smoothscale"):
    #     scaled_image = pygame.transform.smoothscale(i, (w, h))
    # else:
    #     scaled_image = pygame.transform.scale(i, (w, h))
    # pygame.image.save(scaled_image, fout)


if __name__ == "__main__":
    scaleit('in.jpg', 'out.jpg', 100, 100)
