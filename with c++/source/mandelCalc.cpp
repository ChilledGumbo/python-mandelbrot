#include "pch.h"
#include <string>
#include "Complex.cpp"
#include <vector>

#define DLLEXPORT extern "C" __declspec(dllexport)

DLLEXPORT float* getMandelArray(int maxIterations, float bound, double dx, double dy, int width, int height, double xmin, double ymin, float* pixels) {

	double y = ymin;

	int len = width * height;

	for (int j = 0; j < height; j++) {
	
		double x = xmin;

		for (int i = 0; i < width; i++) {
		
			float conv = Complex(x, y).getIterations(maxIterations, bound);

			if (conv == maxIterations) {
			
				pixels[i + j * width] = 0;
			
			}
			else {
			
				pixels[i + j * width] = conv;
			
			}

			x += dx;

		}

		y += dy;
	
	}

	return pixels;

}