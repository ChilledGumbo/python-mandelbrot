from numba import cuda
from numba import *
import numpy as np
from ctypes import *
from numpy.ctypeslib import ndpointer
from pathlib import Path
import matplotlib.pyplot as plt
from decimal import *
from PIL import Image
import pyglet
from pyglet.gl import *
from pyglet.window import mouse


@cuda.jit(device=True)
def mandel(x, y, maxIterations, bound):

    c = complex(x, y)

    z = complex(0,0)

    iterations = 0

    while iterations < maxIterations:

        z = z * z + c

        if (z.real * z.real + z.imag * z.imag) > bound:

            return iterations

        iterations += 1

    return maxIterations

class Mandelbrot:

    def __init__(self, width, height):


        self.width = width
        self.height = height

        self.maxIterations = 255
        self.bound = 4

        w = 2
        h = (w * self.height) / self.width

        self.xmin = -1.5
        self.ymin = -h/2

        self.xmax = self.xmin + w
        self.ymax = self.ymin + h

        self.dx = (self.xmax - self.xmin) / self.width
        self.dy = (self.ymax - self.ymin) / self.height

        self.normalizev = np.vectorize(self.normalize)

        self.image = np.zeros((self.height, self.width), dtype=np.float32)


    def normalize(self, val):

        return int((val / self.maxIterations) * 255)

    
    
    @cuda.jit("void(float32[:,:], float64, float64, float64, float64, int32, int32, int32, float32)")
    def mandel_kernel(image, xmin, ymin, dx, dy, width, height, maxIterations, bound):

        startX, startY = cuda.grid(2)

        gridX = cuda.gridDim.x * cuda.blockDim.x
        gridY = cuda.gridDim.y * cuda.blockDim.y

        for x in range(startX, width, gridX):

            real = xmin + x * dx

            for y in range(startY, height, gridY):

                imag = ymin + y * dy
                image[y, x] = mandel(real, imag, maxIterations, bound)


    def draw(self):

        blockdim = (32, 8)
        griddim = (32, 16)

        gpu_image = cuda.to_device(self.image)
        self.mandel_kernel[griddim, blockdim](gpu_image,
                                              self.xmin,
                                              self.ymin,
                                              self.dx,
                                              self.dy,
                                              self.width,
                                              self.height,
                                              self.maxIterations,
                                              self.bound)
        
        out = self.normalizev(gpu_image.copy_to_host().flatten())

        print(out)

        out = (GLubyte * len(out))(*out)

        texture = pyglet.image.ImageData(self.width, self.height, "I", out)

        texture.blit(0, 0)

        

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

        print(f"window width: {self.xmax-self.xmin}")

        print(f"X bounds: {format(self.xmin, '.120g')} to {format(self.xmax, '.120g')}")
        print(f"Y bounds: {format(self.ymin, '.120g')} to {format(self.ymax, '.120g')}")
        #print(f"dx and dy: {format(self.dx, '.120g')} {format(self.dy, '.120g')}")
        print(f"iterations: {self.maxIterations}")

if __name__ == "__main__":

    m = Mandelbrot(1000, 1000)

    window = pyglet.window.Window(1000, 1000)

    @window.event
    def on_draw():
        window.clear()
        m.draw()

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == mouse.LEFT:
            m.zoom((x, y))
        

    pyglet.app.run()

    
