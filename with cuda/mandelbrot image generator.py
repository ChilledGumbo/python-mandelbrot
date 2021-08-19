from numba import cuda
from numba import *
import numpy as np
from ctypes import *
from numpy.ctypeslib import ndpointer
from pathlib import Path
import pygame
import matplotlib.pyplot as plt
from decimal import *
from PIL import Image


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


@cuda.jit
def mandel_kernel(image, xmin, ymin, dx, dy, width, height, maxIterations, bound):

    startX, startY = cuda.grid(2)

    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, width, gridX):

        real = xmin + x * dx

        for y in range(startY, height, gridY):

            imag = ymin + y * dy
            image[y, x] = mandel(real, imag, maxIterations, bound)

width = 70000
height = 70000

image = np.zeros((height, width), dtype=np.uint8)

bound = 4

maxIterations = 1155

#Set the below values to render a specific spot

xmin = -0.458011510907503971434806544493767432868480682373046875
ymin = 0.59868521731791790596588498374330811202526092529296875

xmax = -0.45801151090697966861142731431755237281322479248046875
ymax = 0.59868521731844215327811298266169615089893341064453125


print(format(xmin, '.120g'), format(xmax, '.120g'))

dx = (xmax - xmin) / width
dy = (ymax - ymin) / height

print(format(dx, '.120g'))

blockdim = (32, 8)
griddim = (32, 16)

gpu_image = cuda.to_device(image)
mandel_kernel[griddim, blockdim](gpu_image, xmin, ymin, dx,
                                 dy, width, height, maxIterations,
                                 bound)

out = np.flip(gpu_image.copy_to_host(), axis=0)

im = Image.fromarray(out)
im.save(f"test.png")

