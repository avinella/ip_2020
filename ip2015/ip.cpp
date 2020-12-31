#include "ip.h"
#include "main.h"
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <time.h>



/*
* convolve with a box filter
*/
Image* ip_blur_box (Image* src, int size)
{
    int totSize = pow(size, 2);
    double* kernel = new double[totSize];
    for (size_t i = 0; i < totSize; ++i) {
        kernel[i] = 1 / (totSize * 1.0);
    }
    Image* newImg = ip_convolve(src, size, kernel);
	return newImg;
}


/*
* convolve with a gaussian filter
* Doesn't seem to be as strong as it should be for sigmas < 1
*/
Image* ip_blur_gaussian (Image* src, int size, double sigma)
{
    int totSize = pow(size, 2);
    double* kernel = new double[totSize];
    double sum = 0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int adjusted = (i * size) + j;
            int distX = size/2 - i;
            int distY = size/2 - j;
            double denom = 1 / (2.0*M_PI*pow(sigma, 2));
            double power = (pow(distX, 2) + pow(distY, 2))/(2.0*pow(sigma, 2));
            double gauss = denom * exp(-power);
            kernel[adjusted] = gauss;
            sum += kernel[adjusted];
        }
    }
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int adjusted = (i * size) + j;
            kernel[adjusted] /= sum;
        }
    }
    Image* newImg = ip_convolve(src, size, kernel);
	return newImg;
}


/*
* convolve with a triangle filter
*/
Image* ip_blur_triangle (Image* src, int size)
{
	cerr << "This function is not implemented." << endl;
	return NULL;
}


/*
* interpolate with a black image
*/
Image* ip_brighten (Image* src, double alpha)
{
    // creates a black image
    Image* blk = new Image{src->getWidth(), src->getHeight()};
    for (int i = 0; i < blk->getWidth(); ++i) {
        for (int j = 0; j < blk->getHeight(); ++j) {
            blk -> setPixel_(i, j, RED, 0);
            blk -> setPixel_(i, j, GREEN, 0);
            blk -> setPixel_(i, j, BLUE, 0);
        }
    }
    Image* final = ip_interpolate(src, blk, alpha);
    return final;
}


/*
* shift colors
*/
Image* ip_color_shift(Image* src)
{
    Image* newImg = new Image{src->getWidth(), src->getHeight()};
    for (int i = 0; i < src->getWidth(); ++i) {
        for (int j = 0; j < src->getHeight(); ++j) {
            Pixel currPixel = src -> getPixel_(i, j);
            double red = currPixel.getColor(RED);
            double green = currPixel.getColor(GREEN);
            double blue = currPixel.getColor(BLUE);
            Pixel shifted{green, blue, red};
            newImg -> setPixel_(i, j, shifted);
        }
    }
    return newImg;
}


/*
* use a mask image for a per-pixel alpha value to perform
* interpolation with a second image
*/
Image* ip_composite (Image* src1, Image* src2, 
					 Image* mask)
{
    Image* newImg = new Image{src1->getWidth(), src1->getHeight()};
    Pixel out;
    for (int i = 0; i < newImg -> getWidth(); ++i) {
        for (int j = 0; j < newImg -> getHeight(); ++j) {
            double alpha = mask -> getPixel_(i, j, RED);
            Pixel pix1 = src1 -> getPixel_(i, j);
            Pixel pix2 = src2 -> getPixel_(i, j);
            double red1 = pix1.getColor(RED);
            double green1 = pix1.getColor(GREEN);
            double blue1 = pix1.getColor(BLUE);
            double red2 = pix2.getColor(RED);
            double green2 = pix2.getColor(GREEN);
            double blue2 = pix2.getColor(BLUE);
            
            double redAvg = alpha * red1 + (1-alpha) * red2;
            double greenAvg = alpha * green1 + (1-alpha) * green2;
            double blueAvg = alpha * blue1 + (1-alpha) * blue2;
            
            out.setColor(RED, redAvg);
            out.setColor(GREEN, greenAvg);
            out.setColor(BLUE, blueAvg);
            newImg -> setPixel_(i, j, out);
        }
    }
    return newImg;
}


/*
* interpolate with the average intensity of the src image
*/
Image* ip_contrast (Image* src, double alpha)
{
    // creates a gray image
    Image* gray = new Image{src->getWidth(), src->getHeight()};
    for (int i = 0; i < gray -> getWidth(); ++i) {
        for (int j = 0; j < gray -> getHeight(); ++j) {
            gray -> setPixel_(i, j, RED, 0.5);
            gray -> setPixel_(i, j, GREEN, 0.5);
            gray -> setPixel_(i, j, BLUE, 0.5);
        }
    }
    Image* final = ip_interpolate(src, gray, alpha);
    return final;
}



/*
* convolve an image with a kernel
*/
Image* ip_convolve (Image* src, int size, double* kernel)
{
    Image* newImg = new Image{src -> getWidth(), src -> getHeight()};
    int totSize = pow(size, 2);
    int boxWidth = size/2;
    int coords[totSize][2];
    int i = -boxWidth;
    int j = -boxWidth;
    for (int k = 0; k < totSize; ++k) {
        coords[k][0] = i;
        coords[k][1] = j;
        if (j == boxWidth) {
            j = -boxWidth;
            ++i;
        } else {
            ++j;
        }
    }

    // "safe" area is numRows-boxWidth
    for (int i = 0; i < newImg -> getWidth(); ++i) {
        for (int j = 0; j < newImg -> getHeight(); ++j) {
            double redSum = 0;
            double greenSum = 0;
            double blueSum = 0;
            for (int k = 0; k < totSize; ++k) {
                int adjusted = ((coords[k][0] + boxWidth) * size) + boxWidth + coords[k][1];
                double red = src -> getPixel_(i + coords[k][0], j + coords[k][1], RED);
                double green = src -> getPixel_(i + coords[k][0], j + coords[k][1], GREEN);
                double blue = src -> getPixel_(i + coords[k][0], j + coords[k][1], BLUE);
                if (!(red == 0.0) || !(green == 0.0) || !(blue == 0.0)) {
                    redSum += kernel[adjusted] * red;
                    greenSum += kernel[adjusted] * green;
                    blueSum += kernel[adjusted] * blue;
                }
            }
            if (redSum > 1.0) {
                redSum = 1.0;
            }
            if (greenSum > 1.0) {
                greenSum = 1.0;
            }
            if (blueSum > 1.0) {
                blueSum = 1.0;
            }
            Pixel pix{redSum, greenSum, blueSum};
            newImg -> setPixel_(i, j, pix);
        }
    }
    delete[] kernel;
	return newImg;
}



/*
*  create cropped version of image
*/
Image* ip_crop (Image* src, int x0, int y0, int x1, int y1)
{
    Image* newImg = new Image{x1 - x0, y1 - y0};
    for (int i = 0; i < newImg -> getWidth(); ++i) {
        for (int j = 0; j < newImg -> getHeight(); ++j) {
            Pixel srcPix = src -> getPixel_(i + x0, j + y0);
            newImg -> setPixel_(i, j, srcPix);
        }
    }
    return newImg;
}
/*
* convolve with an edge detection kernel
*/
Image* ip_edge_detect (Image* src)
{
    
    double* kernel = new double[9];
    for (size_t i = 0; i < 9; ++i) {
        kernel[i] = -1;
        if (i == 4) {
            kernel[i] = 8;
        }
    }
    Image* newImg = ip_convolve(src, 3, kernel);
    return newImg;
}


/*
* extract channel of input image
*/
Image* ip_extract (Image* src, int channel)
{
    Image* newImg = new Image{src->getWidth(), src->getHeight()};
    Pixel single{};
    for (int i = 0; i < newImg -> getWidth(); ++i) {
        for (int j = 0; j < newImg -> getHeight(); ++j) {
            Pixel currPixel = src -> getPixel_(i, j);
            if (channel == 0) {
                newImg -> setPixel_(i, j, RED, currPixel.getColor(RED));
                newImg -> setPixel_(i, j, GREEN, 0.0);
                newImg -> setPixel_(i, j, BLUE, 0.0);
            }
            if (channel == 1) {
                newImg -> setPixel_(i, j, RED, 0.0);
                newImg -> setPixel_(i, j, GREEN, currPixel.getColor(GREEN));
                newImg -> setPixel_(i, j, BLUE, 0.0);
            }
            if (channel == 2) {
                newImg -> setPixel_(i, j, RED, 0.0);
                newImg -> setPixel_(i, j, GREEN, 0.0);
                newImg -> setPixel_(i, j, BLUE, currPixel.getColor(BLUE));
            }
        }
    }
    return newImg;
}


/*
* create your own fun warp
* Bulge warp
* Not yet functional
*/
Image* ip_fun_warp (Image* src, int samplingMode,
                    int gaussianFilterSize, double gaussianSigma)
{
    int x;
    int y;
    cout << "center x (int):";
    cin >> x;
    cout << "center y (int):";
    cin >> y;
    Image* newImg = new Image{src->getWidth(), src->getHeight()};
    for (int i = 0; i < newImg -> getWidth(); ++i) {
        for (int j = 0; j < newImg -> getHeight(); ++j) {
//            double xRatio = x / newImg -> getWidth();
//            double yRatio = y / newImg -> getHeight();
            double r = sqrt(pow(i - x, 2) + pow(j - y, 2));
            double a = atan2(i - x, j - y);
            double rn = pow(r, 2) / x;
            Pixel srcPixel;
            if (samplingMode == I_NEAREST){
                //cerr << rn * cos(a) + x << endl;
                srcPixel = ip_resample_nearest(src, rn * cos(a) + x, rn * sin(a) + y);
            }
            if (samplingMode == I_BILINEAR){
                srcPixel = ip_resample_bilinear(src, rn * cos(a) + x, rn * sin(a) + y);
            }
            if (samplingMode == I_GAUSSIAN){
                srcPixel = ip_resample_gaussian(src, rn * cos(a) + x, rn * sin(a) + y, gaussianFilterSize, gaussianSigma);
            }
            newImg -> setPixel_(newImg -> getWidth() - i, j, srcPixel);
        }
    }
    Image* final = ip_rotate(newImg, 90, x, y, samplingMode, gaussianFilterSize, gaussianSigma);
	return final;
}
/*
* create a new image with values equal to the psychosomatic intensities
* of the source image
*/
Image* ip_grey (Image* src)
{
    Image* gImg = new Image{src->getWidth(), src->getHeight()};
    for (int i = 0; i < gImg -> getWidth(); ++i) {
        for (int j = 0; j < gImg -> getHeight(); ++j) {
            Pixel currPixel = src -> getPixel_(i, j);
            double red = currPixel.getColor(RED);
            double green = currPixel.getColor(GREEN);
            double blue = currPixel.getColor(BLUE);
            double avg = (red + green + blue) / 3;
            Pixel grey{avg, avg, avg};
            gImg -> setPixel_(i, j, grey);
        }
    }
	return gImg;
}


/*
*  shift image by dx and dy (modulo width & height)
*/

Image* ip_image_shift (Image* src, double dx, double dy)
{
    Image* newImg = new Image{src->getWidth(), src->getHeight()};
    for (int i = 0; i < newImg -> getWidth(); ++i) {
        for (int j = 0; j < newImg -> getHeight(); ++j) {
            Pixel srcPix = src -> getPixel_(i, j);
            newImg -> setPixel_(fmod((i + dx), src -> getWidth()), fmod((j + dy), src -> getHeight()), srcPix);
        }
    }
    return newImg;
}
/*
* interpolate an image with another image
*/
Image* ip_interpolate (Image* src1, Image* src2, double alpha)
{
    Image* newImg = new Image{src1->getWidth(), src1->getHeight()};
    Pixel out{};
    for (int i = 0; i < newImg -> getWidth(); ++i) {
        for (int j = 0; j < newImg -> getHeight(); ++j) {
            Pixel pix1 = src1 -> getPixel_(i, j);
            Pixel pix2 = src2 -> getPixel_(i, j);
            double red1 = pix1.getColor(RED);
            double green1 = pix1.getColor(GREEN);
            double blue1 = pix1.getColor(BLUE);
            double red2 = pix2.getColor(RED);
            double green2 = pix2.getColor(GREEN);
            double blue2 = pix2.getColor(BLUE);
            
            double redAvg = alpha * red1 + (1-alpha) * red2;
            double greenAvg = alpha * green1 + (1-alpha) * green2;
            double blueAvg = alpha * blue1 + (1-alpha) * blue2;
            
            out.setColor(RED, redAvg);
            out.setColor(GREEN, greenAvg);
            out.setColor(BLUE, blueAvg);
            newImg -> setPixel_(i, j, out);
        }
    }
    return newImg;
}
/*
* invert input image
*/
Image* ip_invert (Image* src)
{
    // creates a fully gray image
    Image* gray = new Image{src->getWidth(), src->getHeight()};
    for (int i = 0; i < gray->getWidth(); ++i) {
        for (int j = 0; j < gray->getHeight(); ++j) {
            gray -> setPixel_(i, j, RED, 0.5);
            gray -> setPixel_(i, j, GREEN, 0.5);
            gray -> setPixel_(i, j, BLUE, 0.5);
        }
    }
    Image* final = ip_interpolate(gray, src, 2);
    return final;
}


/*
* define your own filter
* you need to request any input paraters here, not in control.cpp
*/

Image* ip_misc(Image* src)
{
    int func;
    cout << "What function? (Shift = 0, Sobel Edge Operator = 1)" << endl;
    cin >> func;
    if (func == 0) {
        int dx;
        int dy;
        cout << "double dx" << endl;
        cin >> dx;
        cout << "double dy" << endl;
        cin >> dy;
        return ip_image_shift(src, dx, dy);
    } else if (func == 1) {
        return ip_sobel(src);
    } else {
        return NULL;
    }
}


/**
 * Not implemented
 */
Image* ip_medianFilter(Image* src, int n) {
    //Image* newImg = new Image{src -> getWidth(), src -> getHeight()};

    return NULL;
}


/*
* round each pixel to the nearest value in the new number of bits
*/
Image* ip_quantize_simple (Image* src, int bitsPerChannel)
{
    Image* newImg = new Image{src -> getWidth(), src -> getHeight()};
    for (int i = 0; i < newImg -> getWidth(); ++i) {
        for (int j = 0; j < newImg -> getHeight(); ++j) {
            double red = round(src -> getPixel(i, j, RED) * (pow(2, bitsPerChannel) - 1)) / (pow(2, bitsPerChannel) - 1);
            double green = round(src -> getPixel(i, j, GREEN) * (pow(2, bitsPerChannel) - 1)) / (pow(2, bitsPerChannel) - 1);
            double blue = round(src -> getPixel(i, j, BLUE) * (pow(2, bitsPerChannel) - 1)) / (pow(2, bitsPerChannel) - 1);
            Pixel pix{red, green, blue};
            newImg -> setPixel_(i, j, pix);
        }
    }
	return newImg;
}


/*
* dither each pixel to the nearest value in the new number of bits
* using a static 2x2 matrix
* not functional yet!
*/
Image* ip_quantize_ordered (Image* src, int bitsPerChannel)
{
    const size_t matrixSize = 2;
    double matrix[matrixSize][matrixSize] = {
        {3, 1},
        {0, 2},
    };
    
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            //matrix[i][j] /= (pow(2, bitsPerChannel) - 1);
        }
    }

    Image* newImg = new Image{src -> getWidth(), src -> getHeight()};
    for (int i = 0; i < newImg -> getWidth(); ++i) {
        for (int j = 0; j < newImg -> getHeight(); ++j) {
            int x = i % matrixSize;
            int y = j % matrixSize;
            double red = src -> getPixel(i, j, RED);
            double green = src -> getPixel(i, j, GREEN);
            double blue = src -> getPixel(i, j, BLUE);
            red += (3 - 2 * matrix[x][y])/8;
            green += (3 - 2 * matrix[x][y])/8;
            blue += (3 - 2 * matrix[x][y])/8;
            double newRed = round(red * (pow(2, bitsPerChannel) - 1))/(pow(2, bitsPerChannel) - 1);
            double newGreen = round(green * (pow(2, bitsPerChannel) - 1))/(pow(2, bitsPerChannel) - 1);
            double newBlue = round(blue * (pow(2, bitsPerChannel) - 1))/(pow(2, bitsPerChannel) - 1);
            Pixel pix{newRed, newGreen, newBlue};
            newImg -> setPixel_(i, j, pix);
        }
    }
    return newImg;
}


/*
* dither each pixel to the nearest value in the new number of bits
* using error diffusion
*/
Image* ip_quantize_fs (Image* src, int bitsPerChannel)
{
    /*
     * uses pattern:
     *         X     7/16
     * 3/16   5/16   1/16
     */
    Image* newImg = new Image{src -> getWidth(), src -> getHeight()};
    
    double redErr[newImg -> getHeight()][newImg -> getWidth()];
    double greenErr[newImg -> getHeight()][newImg -> getWidth()];
    double blueErr[newImg -> getHeight()][newImg -> getWidth()];
    cerr << sizeof(redErr)/sizeof(redErr[0])<< endl;
    for (int i = 0; i < newImg -> getHeight(); ++i) {
        for (int j = 0; j < newImg -> getWidth(); ++j) {
            redErr[i][j] = 0;
            greenErr[i][j] = 0;
            blueErr[i][j] = 0;
        }
    }
    
    for (int i = 0; i < newImg -> getHeight(); ++i) {
        for (int j = 0; j < newImg -> getWidth(); ++j) {
            double redError = 0;
            double greenError = 0;
            double blueError = 0;
            int n = pow(2, bitsPerChannel) - 1;
            
            double red = src -> getPixel_(j, i, RED) + redErr[i][j];
            double green = src -> getPixel_(j, i, GREEN) + greenErr[i][j];
            double blue = src -> getPixel_(j, i, BLUE) + blueErr[i][j];
            
            double newRed = round(red * n)/n;
            double newGreen = round(green * n)/n;
            double newBlue = round(blue * n)/n;
            
            redError = (red - newRed) / 16.0;
            greenError = (green - newGreen) / 16.0;
            blueError = (blue - newBlue) / 16.0;
            //cerr << greenError << endl;
            // right
            if (j < newImg -> getWidth() - 1) {
                redErr[i][j + 1] += redError * 7;
                greenErr[i][j + 1] += greenError * 7;
                blueErr[i][j + 1] += blueError * 7;
            }
            // bottom right
            if (j < newImg -> getWidth() - 1 && i < newImg ->getHeight() - 1) {
                redErr[i + 1][j + 1] += redError;
                greenErr[i + 1][j + 1] += greenError;
                blueErr[i + 1][j + 1] += blueError;
            }
            // bottom
            if (i < newImg ->getHeight() - 1) {
                redErr[i + 1][j] += redError * 5;
                greenErr[i + 1][j] += greenError * 5;
                blueErr[i + 1][j] += blueError * 5;
            }
            // bottom left
            if (j > 0 && i < newImg ->getHeight() - 1) {
                redErr[i + 1][j - 1] += redError * 3;
                greenErr[i + 1][j - 1] += greenError * 3;
                blueErr[i + 1][j - 1] += blueError * 3;
            }
            Pixel pix{newRed, newGreen, newBlue};
            newImg -> setPixel_(j, i, pix);
        }
    }
    return newImg;
}

/*
* nearest neighbor sample
*/
Pixel ip_resample_nearest(Image* src, double x, double y) {
	Pixel myPixel = src -> getPixel_(round(x), round(y));

	return myPixel;
}

/*
* bilinear sample
*/

Pixel ip_resample_bilinear(Image* src, double x, double y) {
    Pixel upLeft = src -> getPixel_(floor(x), floor(y));
    Pixel upRight = src -> getPixel_(floor(x) + 1, floor(y));
    Pixel lowLeft = src -> getPixel_(floor(x), floor(y) + 1);
    Pixel lowRight = src -> getPixel_(floor(x) + 1, floor(y) + 1);
    
    // x interpolation
    double xDist = x - floor(x);
    double x1Red;
    double x1Green;
    double x1Blue;
    double x2Red;
    double x2Green;
    double x2Blue;
    x1Red = upLeft.getColor(RED) * (1-xDist) + upRight.getColor(RED) * xDist;
    x1Green = upLeft.getColor(GREEN) * (1-xDist) + upRight.getColor(GREEN) * xDist;
    x1Blue = upLeft.getColor(BLUE) * (1-xDist) + upRight.getColor(BLUE) * xDist;
    x2Red = lowLeft.getColor(RED) * (1-xDist) + lowRight.getColor(RED) * xDist;
    x2Green = lowLeft.getColor(GREEN) * (1-xDist) + lowRight.getColor(GREEN) * xDist;
    x2Blue = lowLeft.getColor(BLUE) * (1-xDist) + lowRight.getColor(BLUE) * xDist;
    
    // y interpolation
    double yDist = y - floor(y);
    double yRed = x1Red * (1-yDist) + x2Red * yDist;
    double yGreen = x1Green * (1-yDist) + x2Green * yDist;
    double yBlue = x1Blue * (1-yDist) + x2Blue * yDist;
    
    Pixel myPixel{yRed, yGreen, yBlue};
	return myPixel;
}

/*
* gaussian sample
* somehow calls threshold?????
*/
Pixel ip_resample_gaussian(Image* src, double x, double y, int size, double sigma)
{
    double redSum = 0;
    double greenSum = 0;
    double blueSum = 0;
    double sum = 0;
    double kernel[size][size];
    for (int i = -size/2; i < size/2; ++i) {
        for (int j = -size/2; j < size/2; ++j) {
            int distX = i;
            int distY = j;
            double denom = 1 / (2.0*M_PI*pow(sigma, 2));
            double power = (pow(distX, 2) + pow(distY, 2))/(2.0*pow(sigma, 2));
            double gauss = denom * exp(-power);
            kernel[i + size/2][j + size/2] = gauss;
            sum += gauss;
        }
    }
    
    for (int i = 0; i < size; ++i) {
        for (int j = -size/2; j < size/2; ++j) {
            kernel[i][j] /= sum;
        }
    }
    
    for (int i = -size/2; i < size/2; ++i) {
        for (int j = -size/2; j < size/2; ++j) {
            Pixel currPixel = src -> getPixel_(round(x) + i, round(y) + j);
            redSum += kernel[i + size/2][j + size/2] * currPixel.getColor(RED);
            greenSum += kernel[i + size/2][j + size/2] * currPixel.getColor(GREEN);
            blueSum += kernel[i + size/2][j + size/2] * currPixel.getColor(BLUE);
        }
    }
    Pixel myPixel{redSum, greenSum, blueSum};
	return myPixel;
}

/*
* rotate image using one of three sampling techniques
*/
Image* ip_rotate (Image* src, double theta, int x, int y, int samplingMode, 
				  int gaussianFilterSize, double gaussianSigma)
{
	Image* newImg = new Image{src->getWidth(), src->getHeight()};
    double angle = theta * M_PI / 180.0;
    double sin = sinf(angle);
    double cos = cosf(angle);
    for (int i = 0; i < newImg -> getWidth(); ++i) {
        for (int j = 0; j < newImg -> getHeight(); ++j) {
            Pixel srcPixel;
            double xDist = i - x;
            double yDist = j - y;

            if (samplingMode == I_NEAREST){
                srcPixel = ip_resample_nearest(src, cos * xDist - sin * yDist + x, sin * xDist + cos * yDist + y);
            }
            if (samplingMode == I_BILINEAR){
                srcPixel = ip_resample_bilinear(src, cos * xDist - sin * yDist + x, sin * xDist + cos * yDist + y);
            }
            if (samplingMode == I_GAUSSIAN){
                srcPixel = ip_resample_gaussian(src, cos * xDist - sin * yDist + x, sin * xDist + cos * yDist + y, gaussianFilterSize, gaussianSigma);
            }
            newImg -> setPixel_(i, j, srcPixel);
        }
    }
	return newImg;
}


/*
* change saturation
*/
Image* ip_saturate (Image* src, double alpha)
{
    // creates a grey version of src
    Image* gr = ip_grey(src);
    Image* final = ip_interpolate(src, gr, alpha);
    return final;
}


/*
* scale image using one of three sampling techniques
*/
Image* ip_scale (Image* src, double xFac, double yFac, int samplingMode, 
				 int gaussianFilterSize, double gaussianSigma)
{
    Image* newImg = new Image{int(src->getWidth() * xFac), int(src->getHeight() * yFac)};
    for (int i = 0; i < newImg -> getWidth(); ++i) {
        for (int j = 0; j < newImg -> getHeight(); ++j) {
            Pixel srcPixel;
            double xSrc = i / xFac;
            double ySrc = j / yFac;
            if (samplingMode == I_NEAREST){
                srcPixel = ip_resample_nearest(src, xSrc, ySrc);
            }
            if (samplingMode == I_BILINEAR){
                srcPixel = ip_resample_bilinear(src, xSrc, ySrc);
            }
            if (samplingMode == I_GAUSSIAN){
                srcPixel = ip_resample_gaussian(src, xSrc, ySrc, gaussianFilterSize, gaussianSigma);
            }
            newImg -> setPixel_(i, j, srcPixel);
        }
    }
	return newImg;
}


Image* ip_sobel (Image* src) {
    double* kernel = new double[9]{-1, 2, -1, 0, 0, 0, 1, 2, 1};
    Image* newImg = ip_convolve(ip_grey(src), 3, kernel);
    kernel = new double[9]{-1, 0, 1, -2, 0, 2, -1, 0, 1};
    Image* newImg2 = ip_convolve(ip_grey(src), 3, kernel);
    
    Image* finalImg = new Image{src->getWidth(), src->getHeight()};
    for (int i = 0; i < finalImg -> getWidth(); ++i) {
        for (int j = 0; j < finalImg -> getHeight(); ++j) {
            double curr1 = newImg -> getPixel_(i, j, RED);
            double curr2 = newImg2 -> getPixel_(i, j, RED);
            double val = sqrt(pow(curr1, 2) + pow(curr2, 2));
            finalImg -> setPixel_(i, j, RED, val);
            finalImg -> setPixel_(i, j, GREEN, val);
            finalImg -> setPixel_(i, j, BLUE, val);
        }
    }
    
    return finalImg;
}

/*
* threshold image
*/
Image* ip_threshold (Image* src, double cutoff)
{
    Image* newImg = new Image{src->getWidth(), src->getHeight()};
    Pixel thresh{};
    for (int i = 0; i < newImg -> getWidth(); ++i) {
        for (int j = 0; j < newImg -> getHeight(); ++j) {
            Pixel currPixel = src -> getPixel_(i, j);
            double red = currPixel.getColor(RED);
            double green = currPixel.getColor(GREEN);
            double blue = currPixel.getColor(BLUE);
            if (red > cutoff) {
                thresh.setColor(RED, 1.0);
            } else {
                thresh.setColor(RED, 0.0);
            }
            if (green > cutoff) {
                thresh.setColor(GREEN, 1.0);
            } else {
                thresh.setColor(GREEN, 0.0);
            }
            if (blue > cutoff) {
                thresh.setColor(BLUE, 1.0);
            } else {
                thresh.setColor(BLUE, 0.0);
            }
            newImg -> setPixel_(i, j, thresh);
        }
    }
    return newImg;
}




