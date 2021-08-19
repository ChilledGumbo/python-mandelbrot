from ctypes import *
from numpy.ctypeslib import ndpointer
import numpy
from pathlib import Path
import pygame

class Mandelbrot:

    def __init__(self, width, height):

        DLLpath = str(Path(__file__).parent.absolute()) + r"/mandelCalc.dll"

        self.engine = cdll.LoadLibrary(DLLpath)

        pygame.init()

        self.width = width
        self.height = height

        self.engine.getMandelArray.restype = ndpointer(dtype=c_float, shape=(self.width * self.height,))

        self.window = pygame.display.set_mode((self.width, self.height))

        self.maxIterations = 100
        self.bound = 4

        w = 3
        h = (w * self.height) / self.width

        self.xmin = -w/2
        self.ymin = -h/2

        self.xmax = self.xmin + w
        self.ymax = self.ymin + h

        self.dx = (self.xmax - self.xmin) / self.width
        self.dy = (self.ymax - self.ymin) / self.height

        self.grayscalev = numpy.vectorize(self.grayscale)

        self.run()



    def grayscale(self, val):

        val = (val / self.maxIterations) * 255

        return int(val) * 16843008

    def draw(self):

        #print("starting calc")

        empty = (c_float * (self.width * self.height))(*[0 for x in range(self.width * self.height)])

        res = self.engine.getMandelArray(c_int(self.maxIterations),
                                         c_float(self.bound),
                                         c_double(self.dx),
                                         c_double(self.dy),
                                         c_int(self.width),
                                         c_int(self.height),
                                         c_double(self.xmin),
                                         c_double(self.ymin),
                                         empty
                                         )
        #print(res)
        
        #print("calc done")
        
        out = res.reshape((self.height, self.width)).transpose()

        #print(out)

        surface = pygame.surfarray.make_surface(out)

        #pygame.surfarray.blit_array(self.window, out)
        self.window.blit(surface, (0,0))
        pygame.display.flip()


    def zoom(self, mousepos):

        mouseX = mousepos[0]
        mouseY = mousepos[1]

        print(mousepos)

        oldxmin = self.xmin
        oldymin = self.ymin

        self.xmin = self.xmin + (mouseX - 100) * self.dx
        self.xmax = oldxmin + (mouseX + 100) * self.dx
        self.ymin = self.ymin + (mouseY - 100) * self.dy
        self.ymax = oldymin + (mouseY + 100) * self.dy

        self.dx = (self.xmax - self.xmin) / self.width
        self.dy = (self.ymax - self.ymin) / self.height

        self.maxIterations += 50
        

    def run(self):

        running = True

        while running:

            print("running")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.zoom(pygame.mouse.get_pos())

            self.draw()

        pygame.quit()

if __name__ == "__main__":

    m = Mandelbrot(1000, 1000)
