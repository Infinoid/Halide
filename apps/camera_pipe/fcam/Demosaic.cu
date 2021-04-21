#include <Halide.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <map>
#include <vector>
#include <cstring>
#include <cstdio>

// #define DUMP_INPUT
// #define DUMP_DENOISED
// #define DUMP_DEINTERLEAVED
// #define DUMP_DEMOSAICG
// #define DUMP_DEMOSAICED
// #define DUMP_CORRECTED
// #define DUMP_CURVED

using std::cout;
using std::endl;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// #ifdef FCAM_ARCH_ARM
// #include "Demosaic_ARM.h"
// #endif

#include "Demosaic.h"
// #include <FCam/Sensor.h>
// #include <FCam/Time.h>

template<class T>
__device__ T lookup2d(T *in, int x, int y, int width, int height) {
    if(x < 0) {
        x = -x;
    }
    if(y < 0) {
        y = -y;
    }
    if(x >= width) {
        x = width-(x-width-2);
    }
    if(y >= height) {
        y = height-(y-height-2);
    }
    return in[x + y*width];
}

template<class T>
__device__ T lookup3d(T *in, int x, int y, int z, int width, int height, int depth) {
    if(x < 0) {
        x = -x;
    }
    if(y < 0) {
        y = -y;
    }
    if(z < 0) {
        z = -z;
    }
    if(x >= width) {
        x = width-(x-width-2);
    }
    if(y >= height) {
        y = height-(y-height-2);
    }
    if(z >= depth) {
        z = depth-(z-depth-2);
    }
    return in[x + y*width + z*width*height];
}

template<class T>
__device__ T clamp(T value, T lower, T upper) {
    if(value < lower) {
        value = lower;
    }
    if(value > upper) {
        value = upper;
    }
    return value;
}

// stage 1.5
// CameraPipe::hot_pixel_suppression
__global__ void denoise_cuda(uint16_t *in, int inX, int inY, int inoffsetX, int inoffsetY, uint16_t *out, int outX, int outY) {

    // Expr a = max(input(x - 2, y), input(x + 2, y),
    //              input(x, y - 2), input(x, y + 2));

    // Func denoised{"denoised"};
    // denoised(x, y) = clamp(input(x, y), 0, a);

    // return denoised;

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    uint16_t a = max(
        max(
            lookup2d(in, x+inoffsetX - 2, y+inoffsetY, inX, inY),
            lookup2d(in, x+inoffsetX + 2, y+inoffsetY, inX, inY)
        ),
        max(
            lookup2d(in, x+inoffsetX, y+inoffsetY - 2, inX, inY),
            lookup2d(in, x+inoffsetX, y+inoffsetY + 2, inX, inY)
        )
    );

    out[x + y*outX] = clamp<uint16_t>(lookup2d(in, x+inoffsetX, y+inoffsetY, inX, inY), 0, a);
}

// CameraPipe::deinterleave
__global__ void deinterleave_cuda(uint16_t *in, int inX, int inY, uint16_t *out, int outX, int outY, int outZ) {

    // deinterleaved(x, y, c) = mux(c,
    //                              {raw(2 * x, 2 * y),
    //                               raw(2 * x + 1, 2 * y),
    //                               raw(2 * x, 2 * y + 1),
    //                               raw(2 * x + 1, 2 * y + 1)});

    assert(inX >= outX * 2);
    assert(inY >= outY * 2);
    assert(outZ == 4);

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    out[x + y*outX + 0*outX*outY] = lookup2d(in, x*2+0, y*2+0, inX, inY);
    out[x + y*outX + 1*outX*outY] = lookup2d(in, x*2+1, y*2+0, inX, inY);
    out[x + y*outX + 2*outX*outY] = lookup2d(in, x*2+0, y*2+1, inX, inY);
    out[x + y*outX + 3*outX*outY] = lookup2d(in, x*2+1, y*2+1, inX, inY);
}

// Demosaic stage 1
__global__ void demosaic_g_cuda(uint16_t *in, int inX, int inY, int inZ, int16_t *g_r_out, int16_t *g_b_out, int outX, int outY) {
    // interpolate green values at the positions of non-green receptors
    // this produces intermediate buffers to be used in demosaic_rgb_cuda, below
    assert(inX >= outX);
    assert(inY >= outY);
    assert(inZ == 4);

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    // mosaic pattern:
    // Green  Red
    // Blue   Green

    // these are the elements in my mosaic
    int16_t gr_in = lookup3d(in, x, y, 0, inX, inY, inZ); // "gr" == "green in the red row"
    int16_t gb_in = lookup3d(in, x, y, 3, inX, inY, inZ); // "gb" == "green in the blue row"

    // these are the elements in my neighbors' mosaics
    int16_t gr_below = lookup3d(in, x, y+1, 0, inX, inY, inZ);
    int16_t gr_right = lookup3d(in, x+1, y, 0, inX, inY, inZ);

    int16_t gb_above = lookup3d(in, x, y-1, 3, inX, inY, inZ);
    int16_t gb_left  = lookup3d(in, x-1, y, 3, inX, inY, inZ);

    // interpolate the green value that would have been present at the upper right red pixel
    int16_t g_r;
    int16_t gv_r = __hadd(gb_in, gb_above); // vertical
    int16_t gvb_r = abs(gb_in - gb_above);  // vertical delta
    int16_t gh_r = __hadd(gr_in, gr_right); // horizontal
    int16_t ghb_r = abs(gr_in - gr_right);  // horizontal delta
    // pick the one with the smallest delta
    if(gvb_r < ghb_r) {
        g_r = gv_r;
    } else {
        g_r = gh_r;
    }

    // interpolate the green value that would have been present at the lower left blue pixel
    int16_t g_b;
    int16_t gv_b = __hadd(gr_in, gr_below); // vertical
    int16_t gvb_b = abs(gr_in - gr_below);  // vertical delta
    int16_t gh_b = __hadd(gb_in, gb_left); // horizontal
    int16_t ghb_b = abs(gb_in - gb_left);  // horizontal delta
    // pick the one with the smallest delta
    if(gvb_b < ghb_b) {
        g_b = gv_b;
    } else {
        g_b = gh_b;
    }

    g_r_out[x + y*outX] = g_r;
    g_b_out[x + y*outX] = g_b;
}

// Demosaic stage 2
__global__ void demosaic_rgb_cuda(uint16_t *in, int16_t *g_r_in, int16_t *g_b_in, int inX, int inY, int inZ, int16_t *out, int outX, int outY, int outZ) {
    // interpolate red and blue values at the positions of other receptors
    // requires green values to be calculated globally and passed in, see demosaic_g_cuda
    // writes all 3 colors to output
    assert(inX >= outX/2);
    assert(inY >= outY/2);
    assert(inZ == 4);
    assert(outZ == 3);

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    // mosaic pattern:
    // Green  Red
    // Blue   Green

    // these are the elements in my mosaic
    int16_t gr_in = lookup3d(in, x, y, 0, inX, inY, inZ); // "gr" == "green in the red row"
    int16_t r_in  = lookup3d(in, x, y, 1, inX, inY, inZ);
    int16_t b_in  = lookup3d(in, x, y, 2, inX, inY, inZ);
    int16_t gb_in = lookup3d(in, x, y, 3, inX, inY, inZ); // "gb" == "green in the blue row"

    // these are the elements in my neighbors' mosaics
    int16_t r_below  = lookup3d(in, x, y+1, 1, inX, inY, inZ);
    int16_t r_left   = lookup3d(in, x-1, y, 1, inX, inY, inZ);
    int16_t r_lower_left = lookup3d(in, x-1, y+1, 1, inX, inY, inZ);

    int16_t b_above  = lookup3d(in, x, y-1, 2, inX, inY, inZ);
    int16_t b_right  = lookup3d(in, x+1, y, 2, inX, inY, inZ);
    int16_t b_upper_right = lookup3d(in, x+1, y-1, 2, inX, inY, inZ);

    // these are interpolated green values from demosaic_g_cuda, above
    int16_t g_r = lookup2d(g_r_in, x, y, inX, inY);
    int16_t g_r_below = lookup2d(g_r_in, x, y+1, inX, inY);
    int16_t g_r_left  = lookup2d(g_r_in, x-1, y, inX, inY);
    int16_t g_r_lower_left = lookup2d(g_r_in, x-1, y+1, inX, inY);
    int16_t g_b = lookup2d(g_b_in, x, y, inX, inY);
    int16_t g_b_above = lookup2d(g_b_in, x, y-1, inX, inY);
    int16_t g_b_right = lookup2d(g_b_in, x+1, y, inX, inY);
    int16_t g_b_upper_right = lookup2d(g_b_in, x+1, y-1, inX, inY);

    int16_t correction;

    // interpolate the red value that would have been present at the upper left green pixel
    correction = gr_in - __hadd(g_r, g_r_left);
    int16_t r_gr = correction + __hadd(r_in, r_left);

    // interpolate the blue value that would have been present at the upper left green pixel
    correction = gr_in - __hadd(g_b, g_b_above);
    int16_t b_gr = correction + __hadd(b_in, b_above);

    // interpolate the red value that would have been present at the lower right green pixel
    correction = gb_in - __hadd(g_r, g_r_below);
    int16_t r_gb = correction + __hadd(r_in, r_below);

    // interpolate the blue value that would have been present at the lower right green pixel
    correction = gb_in - __hadd(g_b, g_b_right);
    int16_t b_gb = correction + __hadd(b_in, b_right);


    // interpolate the red value that would have been present at the lower left blue pixel
    correction = g_b - __hadd(g_r, g_r_lower_left);
    int16_t rp_b = correction + __hadd(r_in, r_lower_left);
    int16_t rpd_b = abs(r_in - r_lower_left);

    correction = g_b - __hadd(g_r_left, g_r_below);
    int16_t rn_b = correction + __hadd(r_left, r_below);
    int16_t rnd_b = abs(r_left - r_below);

    int16_t r_b;
    if(rpd_b < rnd_b) {
        r_b = rp_b;
    } else {
        r_b = rn_b;
    }

    // interpolate the blue value that would have been present at the upper right red pixel
    correction = g_r - __hadd(g_b, g_b_upper_right);
    int16_t bp_r = correction + __hadd(b_in, b_upper_right);
    int16_t bpd_r = abs(b_in - b_upper_right);

    correction = g_r - __hadd(g_b_right, g_b_above);
    int16_t bn_r = correction + __hadd(b_right, b_above);
    int16_t bnd_r = abs(b_right - b_above);

    int16_t b_r;
    if(bpd_r < bnd_r) {
        b_r = bp_r;
    } else {
        b_r = bn_r;
    }

    // r = interleave_y("interleave_r", interleave_x("interleave_r_g", r_gr, r_r),
    //                                  interleave_x("interleave_r_b", r_b, r_gb));
    out[(x*2+0) + (y*2+0)*outX + (0)*outX*outY] = r_gr;
    out[(x*2+1) + (y*2+0)*outX + (0)*outX*outY] = r_in;
    out[(x*2+0) + (y*2+1)*outX + (0)*outX*outY] = r_b;
    out[(x*2+1) + (y*2+1)*outX + (0)*outX*outY] = r_gb;
    // g = interleave_y("interleave_g", interleave_x("interleave_g_r", g_gr, g_r),
    //                                  interleave_x("interleave_g_b", g_b, g_gb));
    out[(x*2+0) + (y*2+0)*outX + (1)*outX*outY] = gr_in;
    out[(x*2+1) + (y*2+0)*outX + (1)*outX*outY] = g_r;
    out[(x*2+0) + (y*2+1)*outX + (1)*outX*outY] = g_b;
    out[(x*2+1) + (y*2+1)*outX + (1)*outX*outY] = gb_in;
    // b = interleave_y("interleave_b", interleave_x("interleave_b_r", b_gr, b_r),
    //                                  interleave_x("interleave_b_g", b_b, b_gb));
    out[(x*2+0) + (y*2+0)*outX + (2)*outX*outY] = b_gr;
    out[(x*2+1) + (y*2+0)*outX + (2)*outX*outY] = b_r;
    out[(x*2+0) + (y*2+1)*outX + (2)*outX*outY] = b_in;
    out[(x*2+1) + (y*2+1)*outX + (2)*outX*outY] = b_gb;
}

// stage 10
// CameraPipe::color_correct
__global__ void correct_cuda(int16_t *in, float *colormatrix, int inX, int inY, int inZ, int16_t *out, int outX, int outY, int outZ, int whitelevel) {

    assert(inX >= outX);
    assert(inY >= outY);
    assert(inZ == 3);
    assert(outZ == 3);

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    // upcast to int32
    float R = lookup3d(in, x, y, 0, inX, inY, inZ);
    float G = lookup3d(in, x, y, 1, inX, inY, inZ);
    float B = lookup3d(in, x, y, 2, inX, inY, inZ);

    float Rc = R*colormatrix[ 0] + G*colormatrix[ 1] + B*colormatrix[ 2] + colormatrix[ 3];
    float Gc = R*colormatrix[ 4] + G*colormatrix[ 5] + B*colormatrix[ 6] + colormatrix[ 7];
    float Bc = R*colormatrix[ 8] + G*colormatrix[ 9] + B*colormatrix[10] + colormatrix[11];

    // downcast to int16
    out[x + y*outX + 0*outX*outY] = (int16_t)Rc;
    out[x + y*outX + 1*outX*outY] = (int16_t)Gc;
    out[x + y*outX + 2*outX*outY] = (int16_t)Bc;
}

// stage 11
// CameraPipe::apply_curve
__global__ void curve_cuda(int16_t *in, int inX, int inY, int inZ, uint8_t *lut, int lutlen, uint8_t *out, int outX, int outY, int outZ) {
    // apply the brightness curve

    assert(inX >= outX);
    assert(inY >= outY);
    assert(inZ == 3);
    assert(outZ == 3);

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    int16_t r = lookup3d(in, x, y, 0, inX, inY, inZ);
    int16_t g = lookup3d(in, x, y, 1, inX, inY, inZ);
    int16_t b = lookup3d(in, x, y, 2, inX, inY, inZ);

    // clamp the rgb values to be in the range [0, lutlen)
    int16_t minvalue = 0;
    int16_t maxvalue = lutlen - 1;
    r = clamp(r, minvalue, maxvalue);
    g = clamp(g, minvalue, maxvalue);
    b = clamp(b, minvalue, maxvalue);

    // do the lookup
    uint8_t r8 = lut[r];
    uint8_t g8 = lut[g];
    uint8_t b8 = lut[b];

    out[x + y*outX + 0*outX*outY] = r8;
    out[x + y*outX + 1*outX*outY] = g8;
    out[x + y*outX + 2*outX*outY] = b8;
}

// stage 12
// CameraPipe::sharpen
__global__ void sharpen_cuda(uint16_t *in, int inX, int inY, int inZ, uint16_t *out, int outX, int outY, int outZ) {
    // sharpen step is disabled in initial demo
    // TODO: implement this
}

namespace FCam_CUDA {

// Make a linear luminance -> pixel value lookup table
void makeLUT(float contrast, int blackLevel, int whiteLevel, float gamma, unsigned char *lut) {
    unsigned short minRaw = 0 + blackLevel; //f.platform().minRawValue()+blackLevel;
    unsigned short maxRaw = whiteLevel; //f.platform().maxRawValue();

    for (int i = 0; i <= whiteLevel; i++) {
        lut[i] = 0;
    }

    float invRange = 1.0f/(maxRaw - minRaw);
    float b = 2 - powf(2.0f, contrast/100.0f);
    float a = 2 - 2*b;
    for (int i = minRaw+1; i <= maxRaw; i++) {
        // Get a linear luminance in the range 0-1
        float y = (i-minRaw)*invRange;
        // Gamma correct it
        y = powf(y, 1.0f/gamma);
        // Apply a piecewise quadratic contrast curve
        if (y > 0.5) {
            y = 1-y;
            y = a*y*y + b*y;
            y = 1-y;
        } else {
            y = a*y*y + b*y;
        }
        // Convert to 8 bit and save
        y = std::floor(y * 255 + 0.5f);
        if (y < 0) { y = 0; }
        if (y > 255) { y = 255; }
        lut[i] = (unsigned char)y;
    }
}

// From the Halide camera_pipe's color_correct
void makeColorMatrix(float colorMatrix[], float colorTemp) {
    float alpha = (1.f / colorTemp - 1.f/3200.f) / (1.f/7000.f - 1.f/3200.f);

    colorMatrix[0] = alpha*1.6697f     + (1-alpha)*2.2997f;
    colorMatrix[1] = alpha*-0.2693f    + (1-alpha)*-0.4478f;
    colorMatrix[2] = alpha*-0.4004f    + (1-alpha)*0.1706f;
    colorMatrix[3] = alpha*-42.4346f   + (1-alpha)*-39.0923f;

    colorMatrix[4] = alpha*-0.3576f    + (1-alpha)*-0.3826f;
    colorMatrix[5] = alpha*1.0615f     + (1-alpha)*1.5906f;
    colorMatrix[6] = alpha*1.5949f     + (1-alpha)*-0.2080f;
    colorMatrix[7] = alpha*-37.1158f   + (1-alpha)*-25.4311f;

    colorMatrix[8] = alpha*-0.2175f    + (1-alpha)*-0.0888f;
    colorMatrix[9] = alpha*-1.8751f    + (1-alpha)*-0.7344f;
    colorMatrix[10]= alpha*6.9640f     + (1-alpha)*2.2832f;
    colorMatrix[11]= alpha*-26.6970f   + (1-alpha)*-20.0826f;
}

// Some functions used by demosaic
inline int max(int a, int b) {return a>b ? a : b;}
inline int max(int a, int b, int c, int d) {return max(max(a, b), max(c, d));}
inline int min(int a, int b) {return a<b ? a : b;}

void demosaic(Halide::Runtime::Buffer<uint16_t> input, Halide::Runtime::Buffer<uint8_t> out, float colorTemp, float contrast, bool denoise, int blackLevel, int whiteLevel, float gamma) {
    // cout << "cuda demosaic called" << endl;

    int rawWidth = input.width();
    int rawHeight = input.height();
    int borderWidth = 16;
    int borderHeight = 24;
    int fulloutWidth = rawWidth-borderWidth*2;
    int fulloutHeight = rawHeight-borderHeight*2;
    int myoutWidth = min(fulloutWidth, out.width());
    int myoutHeight = min(fulloutHeight, out.height());
    int myinWidth = myoutWidth + borderWidth*2;
    int myinHeight = myoutHeight + borderHeight*2;

    int local_xmin = out.dim(0).min();
    int local_ymin = out.dim(1).min();

    // Prepare the lookup table
    std::vector<unsigned char> lut;
    lut.resize(whiteLevel+1);
    makeLUT(contrast, blackLevel, whiteLevel, gamma, &lut[0]);

    uint8_t *lut_device;
    gpuErrchk( cudaMalloc(&lut_device, lut.size()) );
    gpuErrchk( cudaMemcpy(lut_device, &lut[0], lut.size(), cudaMemcpyHostToDevice) );

    // Grab the color matrix
    float colorMatrix[12];
    makeColorMatrix(colorMatrix, colorTemp);

    float *colormatrix_device;
    gpuErrchk( cudaMalloc(&colormatrix_device, sizeof(float)*12) );
    gpuErrchk( cudaMemcpy(colormatrix_device, colorMatrix, sizeof(float)*12, cudaMemcpyHostToDevice) );

    // input is 2592x1968 (output.x+32, output.y+48) accessed region is 12-2568, 8-1928 (according to header), or -1935 (according to Load logs)
    // input indexes are shifted by (16,12)
    // denoised accessed region is -2-2624, -2-1924 (14-2640, 14-1940 in original coordinates)
    // deinterleaved accessed region is -1-1313, -1,962, 0-4
    // color_corrected accessed region is 0-2560, 0-1920, 0-3
    // curved accessed region is -1-2562, -1-1922, 0-3
    // processed accessed region is 0-2560, 0-1920, 0-3

    // output is 2560x1920x3

#ifdef DUMP_INPUT
    cout << "dumping input from " << input.dim(0).min() << "," << input.dim(1).min() << " to " << input.dim(0).min()+input.dim(0).extent() << "," << input.dim(1).min()+input.dim(1).extent() << endl;
    for(int y = input.dim(1).min(); y < input.dim(1).min()+input.dim(1).extent(); y++) {
        for(int x = input.dim(0).min(); x < input.dim(0).min()+input.dim(0).extent(); x++) {
            printf("input(%d,%d) → %d\n", x, y, input(x, y));
        }
    }
#endif /* DUMP_INPUT */

    // denoise sizing
    int denoisedWidth  = myoutWidth + borderWidth * 2;
    int denoisedHeight = myoutHeight + borderHeight * 2;
    dim3 denoiseblockcount(denoisedWidth, denoisedHeight);

    // denoise input is the image file we were passed
    uint16_t *input_device;
    int input_size = sizeof(input_device[0]) * myinWidth * myinHeight;
    gpuErrchk( cudaMalloc(&input_device, input_size) );
    gpuErrchk( cudaMemcpy2D(input_device, myinWidth*sizeof(input(0,0,0)),
                            &input(local_xmin, local_ymin, 0), rawWidth*sizeof(input(0,0,0)),
                            myinWidth*sizeof(input(0,0,0)), myinHeight,
                            cudaMemcpyHostToDevice) );

    // denoise output buffer
    uint16_t *denoised_device;
    int denoised_size = sizeof(denoised_device[0]) * denoisedWidth * denoisedHeight;
    gpuErrchk( cudaMalloc(&denoised_device, denoised_size) );

    // denoise
    // cout << "calling denoise_cuda" << endl;
    denoise_cuda<<<denoiseblockcount, 1>>>(input_device, myinWidth, myinHeight, 16, 12, denoised_device, denoisedWidth, denoisedHeight);

    gpuErrchk( cudaFree(input_device) );

#ifdef DUMP_DENOISED
    cout << "dumping denoise output" << endl;
    uint16_t *denoised_host = (uint16_t*)malloc(denoised_size);
    gpuErrchk( cudaMemcpy(denoised_host, denoised_device, denoised_size, cudaMemcpyDeviceToHost) );
    for(int y = 0; y < denoisedHeight; y++) {
        for(int x = 0; x < denoisedWidth; x++) {
            printf("denoised(%d,%d) → %d\n", x, y, denoised_host[y*denoisedWidth + x]);
        }
    }
    free(denoised_host);
#endif /* DUMP_DENOISED */

    // deinterleave sizing
    int deinterleavedWidth  = denoisedWidth  >> 1;
    int deinterleavedHeight = denoisedHeight >> 1;
    int deinterleavedDepth  = 4;
    dim3 deinterleaveblockcount(deinterleavedWidth, deinterleavedHeight);

    // deinterleave input is denoise output

    // deinterleave output buffer
    uint16_t *deinterleaved_device;
    int deinterleaved_size = sizeof(deinterleaved_device[0]) * deinterleavedWidth * deinterleavedHeight * deinterleavedDepth;
    gpuErrchk( cudaMalloc(&deinterleaved_device, deinterleaved_size) );

    // deinterleave
    // cout << "calling deinterleave_cuda" << endl;
    deinterleave_cuda<<<deinterleaveblockcount, 1>>>(denoised_device, denoisedWidth, denoisedHeight, deinterleaved_device, deinterleavedWidth, deinterleavedHeight, deinterleavedDepth);

    gpuErrchk( cudaFree(denoised_device) );

#ifdef DUMP_DEINTERLEAVED
    cout << "dumping deinterleave output" << endl;
    uint16_t *deinterleaved_host = (uint16_t*)malloc(deinterleaved_size);
    gpuErrchk( cudaMemcpy(deinterleaved_host, deinterleaved_device, deinterleaved_size, cudaMemcpyDeviceToHost) );
    for(int z = 0; z < deinterleavedDepth; z++) {
        for(int y = 0; y < deinterleavedHeight; y++) {
            for(int x = 0; x < deinterleavedWidth; x++) {
                printf("deinterleaved(%d,%d,%d) → %d\n", x, y, z, deinterleaved_host[z*deinterleavedWidth*deinterleavedHeight + y*deinterleavedWidth + x]);
            }
        }
    }
    free(deinterleaved_host);
#endif /* DUMP_DEINTERLEAVED */


    // demosaic_g sizing
    int demosaicgWidth  = deinterleavedWidth;
    int demosaicgHeight = deinterleavedHeight;
    dim3 demosaicgblockcount(demosaicgWidth, demosaicgHeight);

    // demosaic_g input is deinterleave output

    // demosaic_g outputs 2 intermediate buffers
    int16_t *demosaicg_g_r_device, *demosaicg_g_b_device;
    int demosaicg_size = sizeof(demosaicg_g_r_device[0]) * demosaicgWidth * demosaicgHeight;
    gpuErrchk( cudaMalloc(&demosaicg_g_r_device, demosaicg_size) );
    gpuErrchk( cudaMalloc(&demosaicg_g_b_device, demosaicg_size) );

    // demosaic_r
    // cout << "calling demosaic_g_cuda" << endl;
    demosaic_g_cuda<<<demosaicgblockcount, 1>>>(deinterleaved_device, deinterleavedWidth, deinterleavedHeight, deinterleavedDepth, demosaicg_g_r_device, demosaicg_g_b_device, demosaicgWidth, demosaicgHeight);

#ifdef DUMP_DEMOSAICG
    cout << "dumping demosaic_g output" << endl;
    uint16_t *demosaicg_g_r_host = (uint16_t*)malloc(demosaicg_size);
    gpuErrchk( cudaMemcpy(demosaicg_g_r_host, demosaicg_g_r_device, demosaicg_size, cudaMemcpyDeviceToHost) );
    uint16_t *demosaicg_g_b_host = (uint16_t*)malloc(demosaicg_size);
    gpuErrchk( cudaMemcpy(demosaicg_g_b_host, demosaicg_g_b_device, demosaicg_size, cudaMemcpyDeviceToHost) );
    for(int y = 0; y < demosaicgHeight; y++) {
        for(int x = 0; x < demosaicgWidth; x++) {
            printf("demosaicg_g_r(%d,%d) → %d\n", x, y, demosaicg_g_r_host[y*demosaicgWidth + x]);
            printf("demosaicg_g_b(%d,%d) → %d\n", x, y, demosaicg_g_b_host[y*demosaicgWidth + x]);
        }
    }
    free(demosaicg_g_r_host);
    free(demosaicg_g_b_host);
#endif /* DUMP_DEMOSAICG */


    // demosaic_rgb sizing
    int demosaicedWidth  = deinterleavedWidth  * 2;
    int demosaicedHeight = deinterleavedHeight * 2;
    int demosaicedDepth = 3;
    // each thread computes a 2x2 chunk of output
    dim3 demosaicblockcount(demosaicedWidth / 2, demosaicedHeight / 2);

    // demosaic_rgb input is deinterleave and demosaic_g output

    // demosaic_rgb output buffer
    int16_t *demosaiced_device;
    int demosaiced_size = sizeof(demosaiced_device[0]) * demosaicedWidth * demosaicedHeight * demosaicedDepth;
    gpuErrchk( cudaMalloc(&demosaiced_device, demosaiced_size) );

    // demosaic_rgb
    // cout << "calling demosaic_g_cuda" << endl;
    demosaic_rgb_cuda<<<demosaicblockcount, 1>>>(deinterleaved_device, demosaicg_g_r_device, demosaicg_g_b_device, deinterleavedWidth, deinterleavedHeight, deinterleavedDepth, demosaiced_device, demosaicedWidth, demosaicedHeight, demosaicedDepth);

    gpuErrchk( cudaFree(deinterleaved_device) );
    gpuErrchk( cudaFree(demosaicg_g_r_device) );
    gpuErrchk( cudaFree(demosaicg_g_b_device) );

#ifdef DUMP_DEMOSAICED
    cout << "dumping demosaic_rgb output" << endl;
    uint16_t *demosaiced_host = (uint16_t*)malloc(demosaiced_size);
    gpuErrchk( cudaMemcpy(demosaiced_host, demosaiced_device, demosaiced_size, cudaMemcpyDeviceToHost) );
    for(int z = 0; z < demosaicedDepth; z++) {
        for(int y = 0; y < demosaicedHeight; y++) {
            for(int x = 0; x < demosaicedWidth; x++) {
                printf("demosaiced(%d,%d,%d) → %d\n", x, y, z, demosaiced_host[z*demosaicedWidth*demosaicedHeight + y*demosaicedWidth + x]);
            }
        }
    }
    free(demosaiced_host);
#endif /* DUMP_DEMOSAICED */


    // correct sizing
    int correctedWidth  = demosaicedWidth;
    int correctedHeight = demosaicedHeight;
    int correctedDepth = 3;
    dim3 correctblockcount(correctedWidth, correctedHeight);

    // correct input is demosaic_rgb output

    // correct output buffer
    int16_t *corrected_device;
    int corrected_size = sizeof(corrected_device[0]) * correctedWidth * correctedHeight * correctedDepth;
    gpuErrchk( cudaMalloc(&corrected_device, corrected_size) );

    // correct
    // cout << "calling correct_cuda" << endl;
    correct_cuda<<<correctblockcount, 1>>>(demosaiced_device, colormatrix_device, demosaicedWidth, demosaicedHeight, demosaicedDepth, corrected_device, correctedWidth, correctedHeight, correctedDepth, whiteLevel);

    gpuErrchk( cudaFree(colormatrix_device) );
    gpuErrchk( cudaFree(demosaiced_device) );

#ifdef DUMP_CORRECTED
    cout << "dumping correct output" << endl;
    int16_t *corrected_host = (int16_t*)malloc(corrected_size);
    gpuErrchk( cudaMemcpy(corrected_host, corrected_device, corrected_size, cudaMemcpyDeviceToHost) );
    for(int z = 0; z < correctedDepth; z++) {
        for(int y = 0; y < correctedHeight; y++) {
            for(int x = 0; x < correctedWidth; x++) {
                printf("corrected(%d,%d,%d) → %d\n", x, y, z, corrected_host[z*correctedWidth*correctedHeight + y*correctedWidth + x]);
            }
        }
    }
    free(corrected_host);
#endif /* DUMP_CORRECTED */


    // curve sizing
    int curvedWidth  = out.dim(0).extent();
    int curvedHeight = out.dim(1).extent();
    int curvedDepth = 3;
    dim3 curveblockcount(curvedWidth, curvedHeight);

    // curve input is correct output

    // curve output buffer
    uint8_t *curved_device;
    int curved_size = sizeof(curved_device[0]) * curvedWidth * curvedHeight * curvedDepth;
    gpuErrchk( cudaMalloc(&curved_device, curved_size) );

    // curve
    // cout << "calling curve_cuda" << endl;
    curve_cuda<<<curveblockcount, 1>>>(corrected_device, correctedWidth, correctedHeight, correctedDepth, lut_device, lut.size(), curved_device, curvedWidth, curvedHeight, curvedDepth);

    gpuErrchk( cudaFree(corrected_device) );
    gpuErrchk( cudaFree(lut_device) );

#ifdef DUMP_CURVED
    cout << "dumping curve output" << endl;
    uint8_t *curved_host = (uint8_t*)malloc(curved_size);
    gpuErrchk( cudaMemcpy(curved_host, curved_device, curved_size, cudaMemcpyDeviceToHost) );
    for(int z = 0; z < curvedDepth; z++) {
        for(int y = 0; y < curvedHeight; y++) {
            for(int x = 0; x < curvedWidth; x++) {
                printf("curved(%d,%d,%d) → %d\n", x, y, z, curved_host[z*curvedWidth*curvedHeight + y*curvedWidth + x]);
            }
        }
    }
    free(curved_host);
#endif /* DUMP_CURVED */

    // printf("output buffer starts at [%d,%d,%d]\n",
    //     out.dim(0).min(), out.dim(1).min(), out.dim(2).min());
    // printf("requested output size: [%d,%d,%d]  produced output size: [%d,%d,%d]\n",
    //     out.dim(0).extent(), out.dim(1).extent(), out.dim(2).extent(),
    //     curvedWidth, curvedHeight, curvedDepth);
    // fflush(stdout);

    assert(out.dim(0).extent() == curvedWidth);
    assert(out.dim(1).extent() == curvedHeight);
    assert(out.dim(2).extent() == curvedDepth);
    gpuErrchk( cudaMemcpy(out.begin(), curved_device, curved_size, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(curved_device) );

}

}
