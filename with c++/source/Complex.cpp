#include "pch.h"
#include <math.h>

class Complex {

	public:

		double real;
		double img;

		Complex(double realparam, double imgparam) {
		
			real = realparam;
			img = imgparam;
		
		}

		Complex multi(Complex b) {
		
			double newreal = real * b.real - img * b.img;
			double newimg = real * b.img + img * b.real;

			return Complex(newreal, newimg);
				
		}

		double abs() {
		
			return sqrt(real * real + img * img);
		
		}

		Complex add(Complex b) {

			double newreal = real + b.real;
			double newimg = img + b.img;

			return Complex(newreal, newimg);

		}

		float getIterations(int maxIterations, double bound) {

			Complex z = Complex(0, 0);

			Complex c = Complex(real, img);

			int iterationCount = 0;

			double prevAbs = 0.0;

			float convergeNum = 0.0;

			while (iterationCount < maxIterations) {

				double abs = z.abs();

				if (abs > bound) {

					float diffLast = abs - prevAbs;
					float diffMax = bound - prevAbs;
					convergeNum = iterationCount + (diffMax / diffLast);
					break;

				}

				z = z.multi(z).add(c);
				iterationCount++;
				prevAbs = abs;


			}

			return convergeNum;

		}
	
};