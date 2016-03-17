#include "Halide.h"
#include <stdint.h>

#ifdef HEXAGON
#define VLEN    (1<<LOG2VLEN)
#define BLOCK   (VLEN/2)
#include "halide-hexagon-setup.h"
#else
#define BLOCK   32
#endif

using namespace Halide;
using namespace Halide::Internal;
IRPrinter irp(std::cerr);

int schedule;

#ifndef FIXPT
#define FIXPT  32        // use 32-bit fixed point
#endif

Var x, y, tx("tx"), ty("ty"), c("c");
Func processed("processed");

// Average two positive values rounding up
Expr avg(Expr a, Expr b) {
    Type wider = a.type().with_bits(a.type().bits() * 2);
    return cast(a.type(), (cast(wider, a) + b + 1)/2);
}

Func hot_pixel_suppression(Func input) {
    // hot pixel maximum
    Expr a = max(max(input(x-2, y), input(x+2, y)),
                 max(input(x, y-2), input(x, y+2)));

    Func denoised;
    denoised(x, y) = clamp(input(x, y), 0, a);

    return denoised;
}

Func interleave_x(Func a, Func b) {
    Func out;
    out(x, y) = select((x%2)==0, a(x/2, y), b(x/2, y));
    return out;
}

Func interleave_y(Func a, Func b) {
    Func out;
    out(x, y) = select((y%2)==0, a(x, y/2), b(x, y/2));
    return out;
}

Func deinterleave(Func raw) {
    // Deinterleave the color channels
    Func deinterleaved;

    deinterleaved(x, y, c) = select(c == 0, raw(2*x, 2*y),
                                    select(c == 1, raw(2*x+1, 2*y),
                                           select(c == 2, raw(2*x, 2*y+1),
                                                  raw(2*x+1, 2*y+1))));
    return deinterleaved;
}

Func demosaic(Func deinterleaved) {
    // These are the values we already know from the input
    // x_y = the value of channel x at a site in the input of channel y
    // gb refers to green sites in the blue rows
    // gr refers to green sites in the red rows

    // Give more convenient names to the four channels we know
    Func r_r, g_gr, g_gb, b_b;
    g_gr(x, y) = deinterleaved(x, y, 0);
    r_r(x, y)  = deinterleaved(x, y, 1);
    b_b(x, y)  = deinterleaved(x, y, 2);
    g_gb(x, y) = deinterleaved(x, y, 3);

    // These are the ones we need to interpolate
    Func b_r, g_r, b_gr, r_gr, b_gb, r_gb, r_b, g_b;

    // First calculate green at the red and blue sites

    // Try interpolating vertically and horizontally. Also compute
    // differences vertically and horizontally. Use interpolation in
    // whichever direction had the smallest difference.
    Expr gv_r  = avg(g_gb(x, y-1), g_gb(x, y));
    Expr gvd_r = absd(g_gb(x, y-1), g_gb(x, y));
    Expr gh_r  = avg(g_gr(x+1, y), g_gr(x, y));
    Expr ghd_r = absd(g_gr(x+1, y), g_gr(x, y));

    g_r(x, y)  = select(ghd_r < gvd_r, gh_r, gv_r);

    Expr gv_b  = avg(g_gr(x, y+1), g_gr(x, y));
    Expr gvd_b = absd(g_gr(x, y+1), g_gr(x, y));
    Expr gh_b  = avg(g_gb(x-1, y), g_gb(x, y));
    Expr ghd_b = absd(g_gb(x-1, y), g_gb(x, y));

    g_b(x, y)  = select(ghd_b < gvd_b, gh_b, gv_b);

    // Next interpolate red at gr by first interpolating, then
    // correcting using the error green would have had if we had
    // interpolated it in the same way (i.e. add the second derivative
    // of the green channel at the same place).
    Expr correction;
    correction = g_gr(x, y) - avg(g_r(x, y), g_r(x-1, y));
    r_gr(x, y) = correction + avg(r_r(x-1, y), r_r(x, y));

    // Do the same for other reds and blues at green sites
    correction = g_gr(x, y) - avg(g_b(x, y), g_b(x, y-1));
    b_gr(x, y) = correction + avg(b_b(x, y), b_b(x, y-1));

    correction = g_gb(x, y) - avg(g_r(x, y), g_r(x, y+1));
    r_gb(x, y) = correction + avg(r_r(x, y), r_r(x, y+1));

    correction = g_gb(x, y) - avg(g_b(x, y), g_b(x+1, y));
    b_gb(x, y) = correction + avg(b_b(x, y), b_b(x+1, y));

    // Now interpolate diagonally to get red at blue and blue at
    // red. Hold onto your hats; this gets really fancy. We do the
    // same thing as for interpolating green where we try both
    // directions (in this case the positive and negative diagonals),
    // and use the one with the lowest absolute difference. But we
    // also use the same trick as interpolating red and blue at green
    // sites - we correct our interpolations using the second
    // derivative of green at the same sites.

    correction = g_b(x, y)  - avg(g_r(x, y), g_r(x-1, y+1));
    Expr rp_b  = correction + avg(r_r(x, y), r_r(x-1, y+1));
    Expr rpd_b = absd(r_r(x, y), r_r(x-1, y+1));

    correction = g_b(x, y)  - avg(g_r(x-1, y), g_r(x, y+1));
    Expr rn_b  = correction + avg(r_r(x-1, y), r_r(x, y+1));
    Expr rnd_b = absd(r_r(x-1, y), r_r(x, y+1));

    r_b(x, y)  = select(rpd_b < rnd_b, rp_b, rn_b);


    // Same thing for blue at red
    correction = g_r(x, y)  - avg(g_b(x, y), g_b(x+1, y-1));
    Expr bp_r  = correction + avg(b_b(x, y), b_b(x+1, y-1));
    Expr bpd_r = absd(b_b(x, y), b_b(x+1, y-1));

    correction = g_r(x, y)  - avg(g_b(x+1, y), g_b(x, y-1));
    Expr bn_r  = correction + avg(b_b(x+1, y), b_b(x, y-1));
    Expr bnd_r = absd(b_b(x+1, y), b_b(x, y-1));

    b_r(x, y)  =  select(bpd_r < bnd_r, bp_r, bn_r);

    // Interleave the resulting channels
    Func r = interleave_y(interleave_x(r_gr, r_r),
                          interleave_x(r_b, r_gb));
    Func g = interleave_y(interleave_x(g_gr, g_r),
                          interleave_x(g_b, g_gb));
    Func b = interleave_y(interleave_x(b_gr, b_r),
                          interleave_x(b_b, b_gb));

    Func output;
    output(x, y, c) = select(c == 0, r(x, y),
                             c == 1, g(x, y),
                                     b(x, y));


    /* THE SCHEDULE */
    if (schedule == 0) {
        // optimized for ARM
        // Compute these in chunks over tiles, vectorized by 8
        g_r.compute_at(processed, tx).vectorize(x, 8);
        g_b.compute_at(processed, tx).vectorize(x, 8);
        r_gr.compute_at(processed, tx).vectorize(x, 8);
        b_gr.compute_at(processed, tx).vectorize(x, 8);
        r_gb.compute_at(processed, tx).vectorize(x, 8);
        b_gb.compute_at(processed, tx).vectorize(x, 8);
        r_b.compute_at(processed, tx).vectorize(x, 8);
        b_r.compute_at(processed, tx).vectorize(x, 8);
        // These interleave in y, so unrolling them in y helps
        output.compute_at(processed, tx)
            .vectorize(x, 8)
            .unroll(y, 2)
            .reorder(c, x, y).bound(c, 0, 3).unroll(c);
    } else if (schedule == 1) {
        // optimized for X86
        // Don't vectorize, because sse is bad at 16-bit interleaving
        g_r.compute_at(processed, tx);
        g_b.compute_at(processed, tx);
        r_gr.compute_at(processed, tx);
        b_gr.compute_at(processed, tx);
        r_gb.compute_at(processed, tx);
        b_gb.compute_at(processed, tx);
        r_b.compute_at(processed, tx);
        b_r.compute_at(processed, tx);
        // These interleave in x and y, so unrolling them helps
        output.compute_at(processed, tx).unroll(x, 2).unroll(y, 2)
            .reorder(c, x, y).bound(c, 0, 3).unroll(c);
    } else if (schedule == 2) {
        // optimized for Hexagon
        // Compute these in chunks over tiles, vectorized by 32
        g_r.compute_at(processed, tx).vectorize(x, 32);
        g_b.compute_at(processed, tx).vectorize(x, 32);
        r_gr.compute_at(processed, tx).vectorize(x, 32);
        b_gr.compute_at(processed, tx).vectorize(x, 32);
        r_gb.compute_at(processed, tx).vectorize(x, 32);
        b_gb.compute_at(processed, tx).vectorize(x, 32);
        r_b.compute_at(processed, tx).vectorize(x, 32);
        b_r.compute_at(processed, tx).vectorize(x, 32);
        // These interleave in y, so unrolling them in y helps
        output.compute_at(processed, tx)
            .vectorize(x, 32)
            .unroll(y, 2)
            .reorder(c, x, y).bound(c, 0, 3).unroll(c);
    } else if (schedule == 3) {
        // optimized for Hexagon
        // Compute these in chunks over tiles, scalar
        g_r.compute_at(processed, tx);
        g_b.compute_at(processed, tx);
        r_gr.compute_at(processed, tx);
        b_gr.compute_at(processed, tx);
        r_gb.compute_at(processed, tx);
        b_gb.compute_at(processed, tx);
        r_b.compute_at(processed, tx);
        b_r.compute_at(processed, tx);
        // These interleave in y, so unrolling them in y helps
        output.compute_at(processed, tx)
            .unroll(y, 2)
            .reorder(c, x, y).bound(c, 0, 3).unroll(c);
    } else if (schedule == 4) {
        // optimized for Hexagon
        // Compute these in chunks over tiles, vectorized by 64
        g_r.compute_at(processed, tx).vectorize(x, 64);
        g_b.compute_at(processed, tx).vectorize(x, 64);
        r_gr.compute_at(processed, tx).vectorize(x, 64);
        b_gr.compute_at(processed, tx).vectorize(x, 64);
        r_gb.compute_at(processed, tx).vectorize(x, 64);
        b_gb.compute_at(processed, tx).vectorize(x, 64);
        r_b.compute_at(processed, tx).vectorize(x, 64);
        b_r.compute_at(processed, tx).vectorize(x, 64);
        // These interleave in y, so unrolling them in y helps
        output.compute_at(processed, tx)
            .vectorize(x, 64)
            .unroll(y, 2)
            .reorder(c, x, y).bound(c, 0, 3).unroll(c);
    } else if (schedule == 5) {
        // optimized for Hexagon
        // Compute these in chunks over tiles, scalar
        g_r.compute_at(processed, tx);
        g_b.compute_at(processed, tx);
        r_gr.compute_at(processed, tx);
        b_gr.compute_at(processed, tx);
        r_gb.compute_at(processed, tx);
        b_gb.compute_at(processed, tx);
        r_b.compute_at(processed, tx);
        b_r.compute_at(processed, tx);
        // These interleave in y, so unrolling them in y helps
        output.compute_at(processed, tx)
            .unroll(y, 2)
            .reorder(c, x, y).bound(c, 0, 3).unroll(c);
    } else if (schedule == 6) {
        // no tiling, vectorized by 32
        g_r.compute_at(processed, tx).vectorize(x, 32);
        g_b.compute_at(processed, tx).vectorize(x, 32);
        r_gr.compute_at(processed, tx).vectorize(x, 32);
        b_gr.compute_at(processed, tx).vectorize(x, 32);
        r_gb.compute_at(processed, tx).vectorize(x, 32);
        b_gb.compute_at(processed, tx).vectorize(x, 32);
        r_b.compute_at(processed, tx).vectorize(x, 32);
        b_r.compute_at(processed, tx).vectorize(x, 32);
        // These interleave in y, so unrolling them in y helps
        output.compute_at(processed, tx)
            .vectorize(x, 32)
            .unroll(y, 2)
            .reorder(c, x, y).bound(c, 0, 3).unroll(c);
    } else if (schedule == 7) {
        // no tiling, vectorized by 64
        g_r.compute_at(processed, tx).vectorize(x, 64);
        g_b.compute_at(processed, tx).vectorize(x, 64);
        r_gr.compute_at(processed, tx).vectorize(x, 64);
        b_gr.compute_at(processed, tx).vectorize(x, 64);
        r_gb.compute_at(processed, tx).vectorize(x, 64);
        b_gb.compute_at(processed, tx).vectorize(x, 64);
        r_b.compute_at(processed, tx).vectorize(x, 64);
        b_r.compute_at(processed, tx).vectorize(x, 64);
        // These interleave in y, so unrolling them in y helps
        output.compute_at(processed, tx)
            .vectorize(x, 64)
            .unroll(y, 2)
            .reorder(c, x, y).bound(c, 0, 3).unroll(c);
    } else if (schedule == 8) {
        // no tiling, scalar
        g_r.compute_at(processed, tx);
        g_b.compute_at(processed, tx);
        r_gr.compute_at(processed, tx);
        b_gr.compute_at(processed, tx);
        r_gb.compute_at(processed, tx);
        b_gb.compute_at(processed, tx);
        r_b.compute_at(processed, tx);
        b_r.compute_at(processed, tx);
        // These interleave in y, so unrolling them in y helps
        output.compute_at(processed, tx)
            .unroll(y, 2)
            .reorder(c, x, y).bound(c, 0, 3).unroll(c);
    } else if (schedule == 9) {
        // optimized for Hexagon 64 byte mode.
        // Compute these in chunks over tiles, vectorized by 32
        g_r.compute_at(processed, tx).vectorize(x, 32);
        g_b.compute_at(processed, tx).vectorize(x, 32);
        r_gr.compute_at(processed, tx).vectorize(x, 32);
        b_gr.compute_at(processed, tx).vectorize(x, 32);
        r_gb.compute_at(processed, tx).vectorize(x, 32);
        b_gb.compute_at(processed, tx).vectorize(x, 32);
        r_b.compute_at(processed, tx).vectorize(x, 32);
        b_r.compute_at(processed, tx).vectorize(x, 32);
        // These interleave in y, so unrolling them in y helps
        output.compute_at(processed, tx)
          .unroll(x, 2)
          .unroll(y, 2)
          .reorder(c, x, y).bound(c, 0, 3).unroll(c);
    } else if (schedule == 10) {
        // optimized for Hexagon
        // Compute these in chunks over tiles, vectorized by 64
      g_r.compute_at(processed, tx).vectorize(x, 64);
      g_b.compute_at(processed, tx).vectorize(x, 64);
      r_gr.compute_at(processed, tx).vectorize(x, 64);
      b_gr.compute_at(processed, tx).vectorize(x, 64);
      r_gb.compute_at(processed, tx).vectorize(x, 64);
      b_gb.compute_at(processed, tx).vectorize(x, 64);
      r_b.compute_at(processed, tx).vectorize(x, 64);
      b_r.compute_at(processed, tx).vectorize(x, 64);
      // These interleave in y, so unrolling them in y helps
      output.compute_at(processed, tx)
        .unroll (x, 2)
        .unroll(y, 2)
        .reorder(c, x, y).bound(c, 0, 3).unroll(c);
    } else {
        // Basic naive schedule
        g_r.compute_root();
        g_b.compute_root();
        r_gr.compute_root();
        b_gr.compute_root();
        r_gb.compute_root();
        b_gb.compute_root();
        r_b.compute_root();
        b_r.compute_root();
        output.compute_root();
    }
    return output;
}


Func color_correct(Func input, ImageParam matrix_3200, ImageParam matrix_7000, Param<float> kelvin) {
    // Get a color matrix by linearly interpolating between two
    // calibrated matrices using inverse kelvin.

    Func matrix;
    Expr alpha = (1.0f/kelvin - 1.0f/3200) / (1.0f/7000 - 1.0f/3200);
    Expr val =  (matrix_3200(x, y) * alpha + matrix_7000(x, y) * (1 - alpha));
#if (FIXPT == 32)
    matrix(x, y) = cast<int32_t>(val * 256.0f); // Q8.8 fixed point (32b Q24.8)
#else
    matrix(x, y) = cast<int16_t>(val * 8.0f);   // Q13.3 fixed point in 16b
#endif
    matrix.compute_root();

    Func corrected;
#if (FIXPT == 32)
    Expr ir = cast<int32_t>(input(x, y, 0));
    Expr ig = cast<int32_t>(input(x, y, 1));
    Expr ib = cast<int32_t>(input(x, y, 2));
#else
    Expr ir = cast<int16_t>(input(x, y, 0));
    Expr ig = cast<int16_t>(input(x, y, 1));
    Expr ib = cast<int16_t>(input(x, y, 2));
#endif

    Expr r = matrix(3, 0) + matrix(0, 0) * ir + matrix(1, 0) * ig + matrix(2, 0) * ib;
    Expr g = matrix(3, 1) + matrix(0, 1) * ir + matrix(1, 1) * ig + matrix(2, 1) * ib;
    Expr b = matrix(3, 2) + matrix(0, 2) * ir + matrix(1, 2) * ig + matrix(2, 2) * ib;

#if (FIXPT == 32)
    r = cast<int16_t>(r/256);
    g = cast<int16_t>(g/256);
    b = cast<int16_t>(b/256);
#else
    r = cast<int16_t>(r >> 3);
    g = cast<int16_t>(g >> 3);
    b = cast<int16_t>(b >> 3);
#endif
    corrected(x, y, c) = select(c == 0, r,
                                select(c == 1, g, b));

    return corrected;
}

<<<<<<< HEAD
#ifndef FCAMLUT
=======

>>>>>>> master
Func apply_curve(Func input, Type result_type, Param<float> gamma, Param<float> contrast, Param<int> blackLevel, Param<int> whiteLevel) {
    // copied from FCam
    Func curve("curve");

<<<<<<< HEAD
#ifdef OLD_CURVE
    Expr xf = clamp(cast<float>(x)/1024.0f, 0.0f, 1.0f);
    Expr g = pow(xf, 1.0f/gamma);
=======
    Expr minRaw = 0 + blackLevel;
    Expr maxRaw = whiteLevel;

    Expr invRange = 1.0f/(maxRaw - minRaw);
>>>>>>> master
    Expr b = 2.0f - pow(2.0f, contrast/100.0f);
    Expr a = 2.0f - 2.0f*b;

    // Get a linear luminance in the range 0-1
    Expr xf = clamp(cast<float>(x - minRaw)*invRange, 0.0f, 1.0f);
    // Gamma correct it
    Expr g = pow(xf, 1.0f/gamma);
    // Apply a piecewise quadratic contrast curve
    Expr z = select(g > 0.5f,
                    1.0f - (a*(1.0f-g)*(1.0f-g) + b*(1.0f-g)),
                    a*g*g + b*g);

    // Convert to 8 bit and save
    Expr val = cast(result_type, clamp(z*255.0f+0.5f, 0.0f, 255.0f));
    // makeLUT add guard band outside of (minRaw, maxRaw]:
    curve(x) = select(x <= minRaw, 0, select(x > maxRaw, 255, val));

    curve.compute_root(); // It's a LUT, compute it once ahead of time.

    Func curved;
<<<<<<< HEAD
    curved(x, y, c) = curve(input(x, y, c));
#else // NEW_CURVE (from FCam makeLUT)
    Expr minRaw = 0 + blackLevel;
    Expr maxRaw = whiteLevel;

    Expr invRange = 1.0f/(maxRaw - minRaw);
    Expr b = 2.0f - pow(2.0f, contrast/100.0f);
    Expr a = 2.0f - 2.0f*b;

    // Get a linear luminance in the range 0-1
    Expr xf = clamp(cast<float>(x - minRaw)*invRange, 0.0f, 1.0f);
    // Gamma correct it
    Expr g = pow(xf, 1.0f/gamma);
    // Apply a piecewise quadratic contrast curve
    Expr z = select(g > 0.5f,
                    1.0f - (a*(1.0f-g)*(1.0f-g) + b*(1.0f-g)),
                    a*g*g + b*g);

    // Convert to 8 bit and save
    Expr val = cast(result_type, clamp(z*255.0f+0.5f, 0.0f, 255.0f));
    // makeLUT add guard band outside of (minRaw, maxRaw]:
    curve(x) = select(x <= minRaw, 0, select(x > maxRaw, 255, val));

    curve.compute_root(); // It's a LUT, compute it once ahead of time.

    Func curved;
    // Use clamp to restrict size of LUT as allocated by compute_root
    // - Clamp to variable whiteLevel
    // curved(x, y, c) = curve(clamp(input(x, y, c), 0, whiteLevel));
    // - Clamp to a constant upper value
    curved(x, y, c) = curve(clamp(input(x, y, c), 0, 1023));
#endif

    return curved;
}
#else
Func apply_curve(Func input, Type result_type, ImageParam lut, Param<int> whiteLevel) {
    Func curved;
    // Use pre-computed LUT
    // Assume input stays within LUT bounds
    // curved(x, y, c) = lut(input(x, y, c));
    // Use clamp to make sure input stays within the LUT
    // - Clamp to variable whiteLevel
    // curved(x, y, c) = lut(clamp(input(x, y, c), 0, whiteLevel));
    // - Clamp to a constant upper value
    curved(x, y, c) = lut(clamp(input(x, y, c), 0, 1023));
=======
    // Use clamp to restrict size of LUT as allocated by compute_root
    curved(x, y, c) = curve(clamp(input(x, y, c), 0, 1023));
>>>>>>> master

    return curved;
}
#endif

Func process(Func raw, Type result_type,
             ImageParam matrix_3200, ImageParam matrix_7000, Param<float> color_temp,
<<<<<<< HEAD
             Param<float> gamma, Param<float> contrast, Param<int> blackLevel, Param<int> whiteLevel
#ifdef FCAMLUT
             , ImageParam lut
#endif
             ) {
=======
             Param<float> gamma, Param<float> contrast, Param<int> blackLevel, Param<int> whiteLevel) {
>>>>>>> master

    Var xi, yi;

    Func denoised = hot_pixel_suppression(raw);
    Func deinterleaved = deinterleave(denoised);
    Func demosaiced = demosaic(deinterleaved);
    Func corrected = color_correct(demosaiced, matrix_3200, matrix_7000, color_temp);
<<<<<<< HEAD
#ifdef FCAMLUT
    // use passed in luminance lut
    Func curved = apply_curve(corrected, result_type, lut, whiteLevel);
#else
    Func curved = apply_curve(corrected, result_type, gamma, contrast, blackLevel, whiteLevel);
#endif
=======
    Func curved = apply_curve(corrected, result_type, gamma, contrast, blackLevel, whiteLevel);
>>>>>>> master

    processed(tx, ty, c) = curved(tx, ty, c);

    // Schedule
    processed.bound(c, 0, 3); // bound color loop 0-3, properly
    if (schedule == 0) {
        // Compute in chunks over tiles, vectorized by 8
        denoised.compute_at(processed, tx).vectorize(x, 8);
        deinterleaved.compute_at(processed, tx).vectorize(x, 8).reorder(c, x, y).unroll(c);
        corrected.compute_at(processed, tx).vectorize(x, 4).reorder(c, x, y).unroll(c);
        processed.tile(tx, ty, xi, yi, 32, 32).reorder(xi, yi, c, tx, ty);
        processed.parallel(ty);
    } else if (schedule == 1) {
        // Same as above, but don't vectorize (sse is bad at interleaved 16-bit ops)
        denoised.compute_at(processed, tx);
        deinterleaved.compute_at(processed, tx);
        corrected.compute_at(processed, tx);
        processed.tile(tx, ty, xi, yi, 128, 128).reorder(xi, yi, c, tx, ty);
        processed.parallel(ty);
    } else if (schedule == 2) {
        // optimized for Hexagon
        // Compute in chunks over 32x32 tiles, vectorized by 32
        denoised.compute_at(processed, tx).vectorize(x, 32);
        deinterleaved.compute_at(processed, tx).vectorize(x, 32).reorder(c, x, y).unroll(c);
        corrected.compute_at(processed, tx).vectorize(x, 32).reorder(c, x, y).unroll(c);
        processed.tile(tx, ty, xi, yi, 32, 32).reorder(xi, yi, c, tx, ty);
        processed.parallel(ty);
    } else if (schedule == 3) {
        // optimized for Hexagon
        // Compute in chunks over 32x32 tiles, scalar
        denoised.compute_at(processed, tx);
        deinterleaved.compute_at(processed, tx).reorder(c, x, y).unroll(c);
        corrected.compute_at(processed, tx).reorder(c, x, y).unroll(c);
        processed.tile(tx, ty, xi, yi, 32, 32).reorder(xi, yi, c, tx, ty);
        processed.parallel(ty);
    } else if (schedule == 4) {
        // optimized for Hexagon
        // Compute in chunks over 64x64 tiles, vectorized by 64
        denoised.compute_at(processed, tx).vectorize(x, 64);
        deinterleaved.compute_at(processed, tx).vectorize(x, 64).reorder(c, x, y).unroll(c);
        corrected.compute_at(processed, tx).vectorize(x, 64).reorder(c, x, y).unroll(c);
        processed.tile(tx, ty, xi, yi, 64, 64).reorder(xi, yi, c, tx, ty);
        processed.parallel(ty);
    } else if (schedule == 5) {
        // optimized for Hexagon
        // Compute in chunks over 64x64 tiles, scalar
        denoised.compute_at(processed, tx);
        deinterleaved.compute_at(processed, tx).reorder(c, x, y).unroll(c);
        corrected.compute_at(processed, tx).reorder(c, x, y).unroll(c);
        processed.tile(tx, ty, xi, yi, 64, 64).reorder(xi, yi, c, tx, ty);
        processed.parallel(ty);
    } else if (schedule == 6) {
        // no tiling, vectorized by 32
        denoised.compute_at(processed, tx).vectorize(x, 32);
        deinterleaved.compute_at(processed, tx).vectorize(x, 32).reorder(c, x, y).unroll(c);
        corrected.compute_at(processed, tx).vectorize(x, 32).reorder(c, x, y).unroll(c);
        processed.reorder(c, tx, ty);
        processed.parallel(ty);
    } else if (schedule == 7) {
        // no tiling, vectorized by 64
        denoised.compute_at(processed, tx).vectorize(x, 64);
        deinterleaved.compute_at(processed, tx).vectorize(x, 64).reorder(c, x, y).unroll(c);
        corrected.compute_at(processed, tx).vectorize(x, 64).reorder(c, x, y).unroll(c);
        processed.reorder(c, tx, ty);
        processed.parallel(ty);
    } else if (schedule == 8) {
        // no tiling, scalar
        denoised.compute_at(processed, tx);
        deinterleaved.compute_at(processed, tx).reorder(c, x, y).unroll(c);
        corrected.compute_at(processed, tx).reorder(c, x, y).unroll(c);
        processed.reorder(c, tx, ty);
        processed.parallel(ty);
    } else if (schedule == 9) {
        // optimized for Hexagon 64 byte mode.
        // Compute in chunks over 32x32 tiles, vectorized by 32
        denoised.compute_at(processed, tx).vectorize(x, 32);
        deinterleaved.compute_at(processed, tx).vectorize(x, 32).reorder(c, x, y).unroll(c);
        corrected.compute_at(processed, tx).vectorize(x, 32).reorder(c, x, y).unroll(c);
        processed.tile(tx, ty, xi, yi, 32, 32).reorder(xi, yi, c, tx, ty);
        processed.parallel(ty);
    } else if (schedule == 10) {
        // optimized for Hexagon 128 byte mode.
        // Compute in chunks over 64x64 tiles, vectorized by 64
        denoised.compute_at(processed, tx).vectorize(x, 64);
        deinterleaved.compute_at(processed, tx).vectorize(x, 64).reorder(c, x, y).unroll(c);
        corrected.compute_at(processed, tx).vectorize(x, 64).reorder(c, x, y).unroll(c);
        processed.tile(tx, ty, xi, yi, 64, 64).reorder(xi, yi, c, tx, ty);
        processed.parallel(ty);
    } else {
        denoised.compute_root();
        deinterleaved.compute_root();
        corrected.compute_root();
        processed.compute_root();
    }

    return processed;
}

int main(int argc, char **argv) {
#ifdef HEXAGON
    Target target;
    setupHexagonTarget(target, LOG2VLEN == 7 ? Target::HVX_128 : Target::HVX_64);

    commonPerfSetup(target);
#endif

    // The camera pipe is specialized on the 2592x1968 images that
    // come in, so we'll just use an image instead of a uniform image.
    ImageParam input(UInt(16), 2);
    ImageParam matrix_3200(Float(32), 2, "m3200"), matrix_7000(Float(32), 2, "m7000");
    Param<float> color_temp("color_temp"); //, 3200.0f);
    Param<float> gamma("gamma"); //, 1.8f);
    Param<float> contrast("contrast"); //, 10.0f);
    Param<int> blackLevel("blackLevel"); //, 25);
    Param<int> whiteLevel("whiteLevel"); //, 1023);
<<<<<<< HEAD
#ifdef FCAMLUT
    ImageParam lut(UInt(8), 1, "lut");
#endif
=======
>>>>>>> master

    // shift things inwards to give us enough padding on the
    // boundaries so that we don't need to check bounds. We're going
    // to make a 2560x1920 output image, just like the FCam pipe, so
    // shift by 16, 12
    Func shifted;
    shifted(x, y) = input(x+16, y+12);

    // Parameterized output type, because LLVM PTX (GPU) backend does not
    // currently allow 8-bit computations
    int bit_width = atoi(argv[1]);
    Type result_type = UInt(bit_width);

    // Pick a schedule
    schedule = atoi(argv[2]);

    // Build the pipeline
    Func processed = process(shifted, result_type, matrix_3200, matrix_7000,
<<<<<<< HEAD
                             color_temp, gamma, contrast, blackLevel, whiteLevel
#ifdef FCAMLUT
                             , lut
#endif
                             );
=======
                             color_temp, gamma, contrast, blackLevel, whiteLevel);
>>>>>>> master

    // We can generate slightly better code if we know the output is a whole number of tiles.
    Expr out_width = processed.output_buffer().width();
    Expr out_height = processed.output_buffer().height();
    processed
        .bound(tx, 0, (out_width/BLOCK)*BLOCK)
        .bound(ty, 0, (out_height/BLOCK)*BLOCK);

    //string s = processed.serialize();
    //printf("%s\n", s.c_str());

    std::vector<Argument> args = {color_temp, gamma, contrast, blackLevel, whiteLevel,
<<<<<<< HEAD
                                  input, matrix_3200, matrix_7000
#ifdef FCAMLUT
                                  , lut
#endif
                                  };
#ifdef HEXAGON
    processed.compile_to_file("curved", args, target);
    processed.compile_to_bitcode("curved.bc", args, target);
//    processed.compile_to_assembly("curved.s", args, target);
#else
=======
                                  input, matrix_3200, matrix_7000};
>>>>>>> master
    processed.compile_to_file("curved", args);
    processed.compile_to_assembly("curved.s", args);
#endif

    return 0;
}
