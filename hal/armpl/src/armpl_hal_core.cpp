// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifdef HAVE_ARMPL

#include "armpl_hal_core.hpp"

#include <fftw3.h>
#include <cstring>

#ifdef CV_NEON
#  include <arm_neon.h>
#endif

enum ArmPLDFTMode
{
    ARMPL_DFT_C2C,      // Full 2D complex-to-complex
    ARMPL_DFT_R2C,      // Full 2D real-to-CCS forward
    ARMPL_DFT_C2C_ROW,  // Row-wise complex-to-complex
    ARMPL_DFT_R_ROW     // Row-wise real<->CCS
};

struct ArmPLC2CDFTContext
{
    ArmPLDFTMode mode;
    int          width, height;
    bool         inv, no_scale;
    fftwf_plan   plan_fwd, plan_inv;
    float        scale;

    ArmPLC2CDFTContext()
        : mode(ARMPL_DFT_C2C), width(0), height(0), inv(false), no_scale(true),
          plan_fwd(0), plan_inv(0), scale(1.f) {}
};

struct ArmPLR2CDFTContext
{
    ArmPLDFTMode mode;
    int          width, height;
    bool         col_wise, no_scale;
    fftwf_plan   plan;
    float        scale;

    ArmPLR2CDFTContext()
        : mode(ARMPL_DFT_R2C), width(0), height(0), col_wise(false), no_scale(true),
          plan(0), scale(1.f) {}
};

struct ArmPLC2CRowDFTContext
{
    ArmPLDFTMode   mode;
    int            width, height;
    bool           inv, no_scale;
    fftwf_plan     plan_fwd, plan_inv;
    float          scale;
    fftwf_complex *fftw_buf;   // pre-alloc: one row (width complex floats)

    ArmPLC2CRowDFTContext()
        : mode(ARMPL_DFT_C2C_ROW), width(0), height(0), inv(false), no_scale(true),
          plan_fwd(0), plan_inv(0), scale(1.f), fftw_buf(0) {}
};

struct ArmPLRRowDFTContext
{
    ArmPLDFTMode   mode;
    int            width, height;
    bool           inv, no_scale;
    fftwf_plan     plan_fwd, plan_inv;
    float          scale;
    float         *fftw_in_r;   // pre-alloc: width floats
    fftwf_complex *fftw_out_c;  // pre-alloc: (width/2+1) complex
    fftwf_complex *fftw_in_c;   // pre-alloc: (width/2+1) complex
    float         *fftw_out_r;  // pre-alloc: width floats

    ArmPLRRowDFTContext()
        : mode(ARMPL_DFT_R_ROW), width(0), height(0), inv(false), no_scale(true),
          plan_fwd(0), plan_inv(0), scale(1.f),
          fftw_in_r(0), fftw_out_c(0), fftw_in_c(0), fftw_out_r(0) {}
};

// =======================================================================
// HAL 2D Init
// =======================================================================
int armpl_hal_dftInit2D(cvhalDFT **context,
                        int width, int height,
                        int depth,
                        int src_channels, int dst_channels,
                        int flags, int nonzero_rows)
{
    if (depth        != CV_32F)       return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (nonzero_rows != 0)            return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if ((size_t)width * height <= 64) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const bool isInverse = (flags & CV_HAL_DFT_INVERSE) != 0;
    const bool isScaled  = (flags & CV_HAL_DFT_SCALE)   != 0;
    const bool isRowWise = (flags & CV_HAL_DFT_ROWS)    != 0;

    // ------------------------------------------------------------------
    // C2C full 2D
    // ------------------------------------------------------------------
    if (!isRowWise && src_channels == 2 && dst_channels == 2)
    {
        const int norm_flag = !isScaled ? 8 : (isInverse ? 2 : 1);
        float scale = 1.0f;
        const float inv_total = 1.0f / (float)(width * height);
        if (isInverse) { if (norm_flag == 1 || norm_flag == 2) scale = inv_total; }
        else           { if (norm_flag == 1) scale = inv_total; }

        const size_t total = (size_t)width * height;
        fftwf_complex *tmp_in  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * total);
        fftwf_complex *tmp_out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * total);
        if (!tmp_in || !tmp_out)
        { if (tmp_in) fftwf_free(tmp_in); if (tmp_out) fftwf_free(tmp_out); return CV_HAL_ERROR_NOT_IMPLEMENTED; }

        fftwf_plan pf = fftwf_plan_dft_2d(height, width, tmp_in, tmp_out, FFTW_FORWARD,  FFTW_ESTIMATE);
        fftwf_plan pi = fftwf_plan_dft_2d(height, width, tmp_in, tmp_out, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftwf_free(tmp_in); fftwf_free(tmp_out);
        if (!pf || !pi) { if (pf) fftwf_destroy_plan(pf); if (pi) fftwf_destroy_plan(pi); return CV_HAL_ERROR_NOT_IMPLEMENTED; }

        ArmPLC2CDFTContext *ctx = new ArmPLC2CDFTContext();
        ctx->width = width; ctx->height = height; ctx->inv = isInverse;
        ctx->no_scale = (scale == 1.0f); ctx->plan_fwd = pf; ctx->plan_inv = pi; ctx->scale = scale;
        *context = reinterpret_cast<cvhalDFT*>(ctx);
        return CV_HAL_ERROR_OK;
    }

    // ------------------------------------------------------------------
    // R2C full 2D (forward only)
    // ------------------------------------------------------------------
    if (!isRowWise && src_channels == 1 && dst_channels == 1)
    {
        if (isInverse) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        const int  norm_flag = !isScaled ? 8 : 1;
        const bool col_wise  = (width == 1);
        float scale = 1.0f;
        if (norm_flag == 1)
            scale = col_wise ? (1.0f / height) : (1.0f / (float)(width * height));

        fftwf_plan plan = 0;
        if (col_wise)
        {
            float *dr = (float*)fftwf_malloc(sizeof(float) * height);
            fftwf_complex *dc = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (height/2 + 1));
            if (!dr || !dc) { if (dr) fftwf_free(dr); if (dc) fftwf_free(dc); return CV_HAL_ERROR_NOT_IMPLEMENTED; }
            plan = fftwf_plan_dft_r2c_1d(height, dr, dc, FFTW_ESTIMATE);
            fftwf_free(dr); fftwf_free(dc);
        }
        else
        {
            float *dr = (float*)fftwf_malloc(sizeof(float) * (size_t)width * height);
            fftwf_complex *dc = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (size_t)height * (width/2 + 1));
            if (!dr || !dc) { if (dr) fftwf_free(dr); if (dc) fftwf_free(dc); return CV_HAL_ERROR_NOT_IMPLEMENTED; }
            plan = fftwf_plan_dft_r2c_2d(height, width, dr, dc, FFTW_ESTIMATE);
            fftwf_free(dr); fftwf_free(dc);
        }
        if (!plan) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        ArmPLR2CDFTContext *ctx = new ArmPLR2CDFTContext();
        ctx->width = width; ctx->height = height; ctx->col_wise = col_wise;
        ctx->no_scale = (scale == 1.0f); ctx->plan = plan; ctx->scale = scale;
        *context = reinterpret_cast<cvhalDFT*>(ctx);
        return CV_HAL_ERROR_OK;
    }

    // ------------------------------------------------------------------
    // C2C row-wise
    // ------------------------------------------------------------------
    if (isRowWise && src_channels == 2 && dst_channels == 2)
    {
        const int norm_flag = !isScaled ? 8 : (isInverse ? 2 : 1);
        float scale = 1.0f;
        const float inv_w = 1.0f / (float)width;
        if (isInverse) { if (norm_flag == 1 || norm_flag == 2) scale = inv_w; }
        else           { if (norm_flag == 1) scale = inv_w; }

        fftwf_complex *fftw_buf = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * width);
        if (!fftw_buf) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        fftwf_plan pf = fftwf_plan_dft_1d(width, fftw_buf, fftw_buf, FFTW_FORWARD,  FFTW_ESTIMATE);
        fftwf_plan pi = fftwf_plan_dft_1d(width, fftw_buf, fftw_buf, FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!pf || !pi)
        { if (pf) fftwf_destroy_plan(pf); if (pi) fftwf_destroy_plan(pi); fftwf_free(fftw_buf); return CV_HAL_ERROR_NOT_IMPLEMENTED; }

        ArmPLC2CRowDFTContext *ctx = new ArmPLC2CRowDFTContext();
        ctx->width = width; ctx->height = height; ctx->inv = isInverse;
        ctx->plan_fwd = pf; ctx->plan_inv = pi;
        ctx->scale = scale; ctx->no_scale = (scale == 1.0f); ctx->fftw_buf = fftw_buf;
        *context = reinterpret_cast<cvhalDFT*>(ctx);
        return CV_HAL_ERROR_OK;
    }

    // ------------------------------------------------------------------
    // R_ROW row-wise real<->CCS
    // ------------------------------------------------------------------
    if (isRowWise && src_channels == 1 && dst_channels == 1)
    {
        const int norm_flag = !isScaled ? 8 : (isInverse ? 2 : 1);
        float scale = 1.0f;
        if (!isInverse) { if (norm_flag == 1) scale = 1.0f / (float)width; }
        else            { if (norm_flag != 8) scale = 1.0f / (float)width; }

        float         *fftw_in_r  = (float*)        fftwf_malloc(sizeof(float)         * width);
        fftwf_complex *fftw_out_c = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (width/2 + 1));
        fftwf_complex *fftw_in_c  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (width/2 + 1));
        float         *fftw_out_r = (float*)        fftwf_malloc(sizeof(float)         * width);
        if (!fftw_in_r || !fftw_out_c || !fftw_in_c || !fftw_out_r)
        {
            if (fftw_in_r)  fftwf_free(fftw_in_r);  if (fftw_out_c) fftwf_free(fftw_out_c);
            if (fftw_in_c)  fftwf_free(fftw_in_c);  if (fftw_out_r) fftwf_free(fftw_out_r);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        fftwf_plan pf = fftwf_plan_dft_r2c_1d(width, fftw_in_r, fftw_out_c, FFTW_ESTIMATE);
        fftwf_plan pi = fftwf_plan_dft_c2r_1d(width, fftw_in_c, fftw_out_r, FFTW_ESTIMATE);
        if (!pf || !pi)
        {
            if (pf) fftwf_destroy_plan(pf); if (pi) fftwf_destroy_plan(pi);
            fftwf_free(fftw_in_r); fftwf_free(fftw_out_c); fftwf_free(fftw_in_c); fftwf_free(fftw_out_r);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        ArmPLRRowDFTContext *ctx = new ArmPLRRowDFTContext();
        ctx->width = width; ctx->height = height; ctx->inv = isInverse;
        ctx->plan_fwd = pf; ctx->plan_inv = pi;
        ctx->scale = scale; ctx->no_scale = (scale == 1.0f);
        ctx->fftw_in_r = fftw_in_r; ctx->fftw_out_c = fftw_out_c;
        ctx->fftw_in_c = fftw_in_c; ctx->fftw_out_r = fftw_out_r;
        *context = reinterpret_cast<cvhalDFT*>(ctx);
        return CV_HAL_ERROR_OK;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =======================================================================
// HAL 2D Execute
// =======================================================================
int armpl_hal_dft2D(cvhalDFT *context,
                    const unsigned char *src_data, size_t src_step,
                    unsigned char       *dst_data, size_t dst_step)
{
    if (!context || !src_data || !dst_data)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const ArmPLDFTMode mode = *reinterpret_cast<const ArmPLDFTMode*>(context);

    if (mode == ARMPL_DFT_C2C)
    {
        ArmPLC2CDFTContext *ctx = reinterpret_cast<ArmPLC2CDFTContext*>(context);
        if (!ctx->plan_fwd || !ctx->plan_inv) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        const int    W        = ctx->width;
        const int    H        = ctx->height;
        const float  sc       = ctx->scale;
        const bool   no_scale = ctx->no_scale;
        const size_t row_cb   = (size_t)W * sizeof(fftwf_complex);

        fftwf_complex *in  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (size_t)W * H);
        fftwf_complex *out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (size_t)W * H);
        if (!in || !out) { if (in) fftwf_free(in); if (out) fftwf_free(out); return CV_HAL_ERROR_NOT_IMPLEMENTED; }

        // OPT-B copy-in
        if (src_step == row_cb)
            memcpy(in, src_data, (size_t)W * H * sizeof(fftwf_complex));
        else
        {
            for (int y = 0; y < H; y++)
            {
                const float   *sr = reinterpret_cast<const float*>(src_data + (size_t)y * src_step);
                fftwf_complex *ir = in + (size_t)y * W;
                int x = 0;
#ifdef CV_NEON
                for (; x + 3 < W; x += 4)
                {
                    vst1q_f32((float*)(ir+x),   vld1q_f32(sr + x*2));
                    vst1q_f32((float*)(ir+x+2), vld1q_f32(sr + (x+2)*2));
                }
#endif
                for (; x < W; x++) { ir[x][0] = sr[x*2]; ir[x][1] = sr[x*2+1]; }
            }
        }

        fftwf_execute_dft(ctx->inv ? ctx->plan_inv : ctx->plan_fwd, in, out);

        // OPT-A + OPT-B copy-out
        if (no_scale)
        {
            if (dst_step == row_cb)
                memcpy(dst_data, out, (size_t)W * H * sizeof(fftwf_complex));
            else
                for (int y = 0; y < H; y++)
                    memcpy(dst_data + (size_t)y * dst_step, out + (size_t)y * W, row_cb);
        }
        else
        {
#ifdef CV_NEON
            const float32x4_t sv = vdupq_n_f32(sc);
#endif
            for (int y = 0; y < H; y++)
            {
                float               *dr = reinterpret_cast<float*>(dst_data + (size_t)y * dst_step);
                const fftwf_complex *or_ = out + (size_t)y * W;
                int x = 0;
#ifdef CV_NEON
                for (; x + 3 < W; x += 4)
                {
                    vst1q_f32(dr + x*2,     vmulq_f32(vld1q_f32((const float*)(or_+x)),   sv));
                    vst1q_f32(dr + (x+2)*2, vmulq_f32(vld1q_f32((const float*)(or_+x+2)), sv));
                }
#endif
                for (; x < W; x++) { dr[x*2] = or_[x][0]*sc; dr[x*2+1] = or_[x][1]*sc; }
            }
        }

        fftwf_free(in); fftwf_free(out);
        return CV_HAL_ERROR_OK;
    }

    if (mode == ARMPL_DFT_R2C)
    {
        ArmPLR2CDFTContext *ctx = reinterpret_cast<ArmPLR2CDFTContext*>(context);
        if (!ctx->plan) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        const int    W        = ctx->width;
        const int    H        = ctx->height;
        const float  sc       = ctx->scale;
        const bool   no_scale = ctx->no_scale;

        // ---- Column-wise 1D (width==1) ----
        if (ctx->col_wise)
        {
            float         *in  = (float*)        fftwf_malloc(sizeof(float)         * H);
            fftwf_complex *out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (H/2 + 1));
            if (!in || !out) { if (in) fftwf_free(in); if (out) fftwf_free(out); return CV_HAL_ERROR_NOT_IMPLEMENTED; }

            for (int y = 0; y < H; y++)
                in[y] = reinterpret_cast<const float*>(src_data + (size_t)y * src_step)[0];
            fftwf_execute_dft_r2c(ctx->plan, in, out);

            const int pairs = (H - 1) / 2;
            if (no_scale)
            {
                reinterpret_cast<float*>(dst_data)[0] = out[0][0];
                for (int k = 1; k <= pairs; k++)
                {
                    reinterpret_cast<float*>(dst_data + (size_t)(2*k-1)*dst_step)[0] = out[k][0];
                    reinterpret_cast<float*>(dst_data + (size_t)(2*k)  *dst_step)[0] = out[k][1];
                }
                if ((H & 1) == 0)
                    reinterpret_cast<float*>(dst_data + (size_t)(H-1)*dst_step)[0] = out[H/2][0];
            }
            else
            {
                reinterpret_cast<float*>(dst_data)[0] = out[0][0] * sc;
                for (int k = 1; k <= pairs; k++)
                {
                    reinterpret_cast<float*>(dst_data + (size_t)(2*k-1)*dst_step)[0] = out[k][0] * sc;
                    reinterpret_cast<float*>(dst_data + (size_t)(2*k)  *dst_step)[0] = out[k][1] * sc;
                }
                if ((H & 1) == 0)
                    reinterpret_cast<float*>(dst_data + (size_t)(H-1)*dst_step)[0] = out[H/2][0] * sc;
            }
            fftwf_free(in); fftwf_free(out);
            return CV_HAL_ERROR_OK;
        }

        // ---- 2D real-to-CCS ----
        float         *in  = (float*)        fftwf_malloc(sizeof(float)         * (size_t)W * H);
        fftwf_complex *out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (size_t)H * (W/2 + 1));
        if (!in || !out) { if (in) fftwf_free(in); if (out) fftwf_free(out); return CV_HAL_ERROR_NOT_IMPLEMENTED; }

        // OPT-C copy-in
        if (src_step == (size_t)W * sizeof(float))
            memcpy(in, src_data, (size_t)W * H * sizeof(float));
        else
            for (int y = 0; y < H; y++)
                memcpy(in + (size_t)y * W,
                       reinterpret_cast<const float*>(src_data + (size_t)y * src_step),
                       (size_t)W * sizeof(float));

        fftwf_execute_dft_r2c(ctx->plan, in, out);

        // OPT-D: CCS packing with hoisted col-0 branch.
        //
        // FFTW r2c output layout: out[y][k], y in [0,H), k in [0,W/2+1)
        // OpenCV CCS row y packing:
        //   col-0: y==0||y==1 → out[y*(W/2+1)][0]
        //          y even     → out[(y/2)*(W/2+1)][1]
        //          y odd      → out[((y+1)/2)*(W/2+1)][0]
        //   cols 2k-1,2k (k>=1): out[y*(W/2+1)+k][0], out[y*(W/2+1)+k][1]
        //   Nyquist (even W): same col-0 rule but for index W/2
        //
        const int half1  = W/2 + 1;
        const int pairs  = (W - 1) / 2;
        const int even_W = (W & 1) == 0;

        // Helper lambda-like macro: pack bins 1..pairs + optional Nyquist for row fi.
        // col-0 is already written by the caller before invoking this.
#define PACK_BINS(dr_, fi_)                                                 \
        do {                                                                \
            float *_d = (dr_); int _fi = (fi_);                            \
            if (no_scale) {                                                 \
                for (int k=1; k<=pairs; k++) {                             \
                    _d[2*k-1] = out[_fi+k][0]; _d[2*k] = out[_fi+k][1];  \
                }                                                           \
            } else {                                                        \
                for (int k=1; k<=pairs; k++) {                             \
                    _d[2*k-1] = out[_fi+k][0]*sc; _d[2*k] = out[_fi+k][1]*sc; \
                }                                                           \
            }                                                               \
        } while(0)

        // y == 0
        if (H > 0)
        {
            float *dr = reinterpret_cast<float*>(dst_data);
            dr[0] = no_scale ? out[0][0] : out[0][0]*sc;
            PACK_BINS(dr, 0);
            if (even_W) dr[W-1] = no_scale ? out[W/2][0] : out[W/2][0]*sc;
        }
        // y == 1
        if (H > 1)
        {
            float *dr = reinterpret_cast<float*>(dst_data + dst_step);
            int fi = half1;
            dr[0] = no_scale ? out[fi][0] : out[fi][0]*sc;
            PACK_BINS(dr, fi);
            if (even_W) dr[W-1] = no_scale ? out[fi+W/2][0] : out[fi+W/2][0]*sc;
        }
        // OPT-D even rows y=2,4,6,... — col-0 = imag of row y/2, branch-free
        for (int y = 2; y < H; y += 2)
        {
            float *dr      = reinterpret_cast<float*>(dst_data + (size_t)y * dst_step);
            int    fi      = y * half1;
            int    fi_c0   = (y/2) * half1;
            dr[0] = no_scale ? out[fi_c0][1] : out[fi_c0][1]*sc;
            PACK_BINS(dr, fi);
            if (even_W) dr[W-1] = no_scale ? out[fi_c0+W/2][1] : out[fi_c0+W/2][1]*sc;
        }
        // OPT-D odd rows y=3,5,7,... — col-0 = real of row (y+1)/2, branch-free
        for (int y = 3; y < H; y += 2)
        {
            float *dr      = reinterpret_cast<float*>(dst_data + (size_t)y * dst_step);
            int    fi      = y * half1;
            int    fi_c0   = ((y+1)/2) * half1;
            dr[0] = no_scale ? out[fi_c0][0] : out[fi_c0][0]*sc;
            PACK_BINS(dr, fi);
            if (even_W) dr[W-1] = no_scale ? out[fi_c0+W/2][0] : out[fi_c0+W/2][0]*sc;
        }

#undef PACK_BINS

        fftwf_free(in); fftwf_free(out);
        return CV_HAL_ERROR_OK;
    }

    // ------------------------------------------------------------------
    // C2C row-wise execute
    //
    // FORWARD (OPT-E): in-place on dst, benchmark-confirmed 1.36×/1.42×.
    // INVERSE: plan's own buffer path (reverted from in-place, was 0.89×).
    // OPT-A: no_scale fast-path in both directions.
    // ------------------------------------------------------------------
    if (mode == ARMPL_DFT_C2C_ROW)
    {
        ArmPLC2CRowDFTContext *ctx = reinterpret_cast<ArmPLC2CRowDFTContext*>(context);
        if (!ctx->plan_fwd || !ctx->plan_inv) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        const int      W        = ctx->width;
        const int      H        = ctx->height;
        const float    sc       = ctx->scale;
        const bool     no_scale = ctx->no_scale;
        const size_t   rb       = (size_t)W * sizeof(fftwf_complex);
        fftwf_complex *buf      = ctx->fftw_buf;

        if (!ctx->inv)
        {
            fftwf_plan plan = ctx->plan_fwd;
            for (int i = 0; i < H; i++)
            {
                const unsigned char *sb = src_data + (size_t)i * src_step;
                unsigned char       *db = dst_data + (size_t)i * dst_step;
                if (sb != db) memcpy(db, sb, rb);
                fftwf_execute_dft(plan,
                                  reinterpret_cast<fftwf_complex*>(db),
                                  reinterpret_cast<fftwf_complex*>(db));
                if (!no_scale)
                {
                    float *f = reinterpret_cast<float*>(db);
                    int j = 0;
#ifdef CV_NEON
                    const float32x4_t sv = vdupq_n_f32(sc);
                    for (; j + 7 < W*2; j += 8)
                    { vst1q_f32(f+j,   vmulq_f32(vld1q_f32(f+j),   sv));
                      vst1q_f32(f+j+4, vmulq_f32(vld1q_f32(f+j+4), sv)); }
#endif
                    for (; j < W*2; j++) f[j] *= sc;
                }
            }
        }
        else
        {
            for (int i = 0; i < H; i++)
            {
                const unsigned char *sb = src_data + (size_t)i * src_step;
                unsigned char       *db = dst_data + (size_t)i * dst_step;
                memcpy(buf, sb, rb);
                fftwf_execute(ctx->plan_inv);
                if (no_scale)
                    memcpy(db, buf, rb);
                else
                {
                    float       *df = reinterpret_cast<float*>(db);
                    const float *bf = reinterpret_cast<const float*>(buf);
                    int j = 0;
#ifdef CV_NEON
                    const float32x4_t sv = vdupq_n_f32(sc);
                    for (; j + 7 < W*2; j += 8)
                    { vst1q_f32(df+j,   vmulq_f32(vld1q_f32(bf+j),   sv));
                      vst1q_f32(df+j+4, vmulq_f32(vld1q_f32(bf+j+4), sv)); }
#endif
                    for (; j < W*2; j++) df[j] = bf[j]*sc;
                }
            }
        }
        return CV_HAL_ERROR_OK;
    }

    // ------------------------------------------------------------------
    // R_ROW row-wise real<->CCS execute
    //
    // FORWARD: memcpy → r2c execute → CCS pack (4-wide unroll, OPT-F)
    // INVERSE: CCS unpack (4-wide unroll) → c2r execute → scale+copy
    // OPT-A: no_scale fast-path in both directions.
    // ------------------------------------------------------------------
    if (mode == ARMPL_DFT_R_ROW)
    {
        ArmPLRRowDFTContext *ctx = reinterpret_cast<ArmPLRRowDFTContext*>(context);
        if (!ctx->plan_fwd || !ctx->plan_inv) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        const int   W        = ctx->width;
        const int   H        = ctx->height;
        const float sc       = ctx->scale;
        const bool  no_scale = ctx->no_scale;

        if (!ctx->inv)
        {
            float         *fin  = ctx->fftw_in_r;
            fftwf_complex *fout = ctx->fftw_out_c;
            const int      ncf  = (W - 1) / 2;   // number of complex freq bins
            const int      hnyq = (W & 1) == 0;  // has Nyquist

            for (int i = 0; i < H; i++)
            {
                const float *sr = reinterpret_cast<const float*>(src_data + (size_t)i * src_step);
                float       *dr = reinterpret_cast<float*>(dst_data + (size_t)i * dst_step);

                memcpy(fin, sr, (size_t)W * sizeof(float));
                fftwf_execute(ctx->plan_fwd);

                if (no_scale)
                {
                    dr[0] = fout[0][0];
                    int j = 1;
                    for (; j+3 <= ncf; j += 4)
                    {
                        dr[j*2-1]     = fout[j][0];   dr[j*2]     = fout[j][1];
                        dr[(j+1)*2-1] = fout[j+1][0]; dr[(j+1)*2] = fout[j+1][1];
                        dr[(j+2)*2-1] = fout[j+2][0]; dr[(j+2)*2] = fout[j+2][1];
                        dr[(j+3)*2-1] = fout[j+3][0]; dr[(j+3)*2] = fout[j+3][1];
                    }
                    for (; j <= ncf; j++) { dr[j*2-1] = fout[j][0]; dr[j*2] = fout[j][1]; }
                    if (hnyq) dr[W-1] = fout[W/2][0];
                }
                else
                {
                    dr[0] = fout[0][0] * sc;
                    int j = 1;
                    for (; j+3 <= ncf; j += 4)
                    {
                        dr[j*2-1]     = fout[j][0]*sc;   dr[j*2]     = fout[j][1]*sc;
                        dr[(j+1)*2-1] = fout[j+1][0]*sc; dr[(j+1)*2] = fout[j+1][1]*sc;
                        dr[(j+2)*2-1] = fout[j+2][0]*sc; dr[(j+2)*2] = fout[j+2][1]*sc;
                        dr[(j+3)*2-1] = fout[j+3][0]*sc; dr[(j+3)*2] = fout[j+3][1]*sc;
                    }
                    for (; j <= ncf; j++) { dr[j*2-1] = fout[j][0]*sc; dr[j*2] = fout[j][1]*sc; }
                    if (hnyq) dr[W-1] = fout[W/2][0] * sc;
                }
            }
        }
        else
        {
            fftwf_complex *fin  = ctx->fftw_in_c;
            float         *fout = ctx->fftw_out_r;
            const bool     hnyq = (W & 1) == 0;

            for (int i = 0; i < H; i++)
            {
                const float *sr = reinterpret_cast<const float*>(src_data + (size_t)i * src_step);
                float       *dr = reinterpret_cast<float*>(dst_data + (size_t)i * dst_step);

                fin[0][0] = sr[0]; fin[0][1] = 0.f;

                if (hnyq)
                {
                    int j = 1, end = W/2;
                    for (; j+3 < end; j += 4)
                    {
                        fin[j][0]=sr[j*2-1];     fin[j][1]=sr[j*2];
                        fin[j+1][0]=sr[(j+1)*2-1]; fin[j+1][1]=sr[(j+1)*2];
                        fin[j+2][0]=sr[(j+2)*2-1]; fin[j+2][1]=sr[(j+2)*2];
                        fin[j+3][0]=sr[(j+3)*2-1]; fin[j+3][1]=sr[(j+3)*2];
                    }
                    for (; j < end; j++) { fin[j][0]=sr[j*2-1]; fin[j][1]=sr[j*2]; }
                    fin[W/2][0] = sr[W-1]; fin[W/2][1] = 0.f;
                }
                else
                {
                    int j = 1, end = W/2+1;
                    for (; j+3 < end; j += 4)
                    {
                        fin[j][0]=sr[j*2-1];     fin[j][1]=sr[j*2];
                        fin[j+1][0]=sr[(j+1)*2-1]; fin[j+1][1]=sr[(j+1)*2];
                        fin[j+2][0]=sr[(j+2)*2-1]; fin[j+2][1]=sr[(j+2)*2];
                        fin[j+3][0]=sr[(j+3)*2-1]; fin[j+3][1]=sr[(j+3)*2];
                    }
                    for (; j < end; j++) { fin[j][0]=sr[j*2-1]; fin[j][1]=sr[j*2]; }
                }

                fftwf_execute(ctx->plan_inv);

                if (no_scale)
                    memcpy(dr, fout, (size_t)W * sizeof(float));
                else
                {
                    int j = 0;
#ifdef CV_NEON
                    const float32x4_t sv = vdupq_n_f32(sc);
                    for (; j+3 < W; j += 4)
                        vst1q_f32(dr+j, vmulq_f32(vld1q_f32(fout+j), sv));
#endif
                    for (; j < W; j++) dr[j] = fout[j]*sc;
                }
            }
        }
        return CV_HAL_ERROR_OK;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int armpl_hal_dftFree2D(cvhalDFT *context)
{
    if (!context) return CV_HAL_ERROR_OK;

    const ArmPLDFTMode mode = *reinterpret_cast<const ArmPLDFTMode*>(context);

    if (mode == ARMPL_DFT_C2C)
    {
        ArmPLC2CDFTContext *ctx = reinterpret_cast<ArmPLC2CDFTContext*>(context);
        if (ctx->plan_fwd) fftwf_destroy_plan(ctx->plan_fwd);
        if (ctx->plan_inv) fftwf_destroy_plan(ctx->plan_inv);
        delete ctx;
    }
    else if (mode == ARMPL_DFT_R2C)
    {
        ArmPLR2CDFTContext *ctx = reinterpret_cast<ArmPLR2CDFTContext*>(context);
        if (ctx->plan) fftwf_destroy_plan(ctx->plan);
        delete ctx;
    }
    else if (mode == ARMPL_DFT_C2C_ROW)
    {
        ArmPLC2CRowDFTContext *ctx = reinterpret_cast<ArmPLC2CRowDFTContext*>(context);
        if (ctx->plan_fwd) fftwf_destroy_plan(ctx->plan_fwd);
        if (ctx->plan_inv) fftwf_destroy_plan(ctx->plan_inv);
        if (ctx->fftw_buf) fftwf_free(ctx->fftw_buf);
        delete ctx;
    }
    else if (mode == ARMPL_DFT_R_ROW)
    {
        ArmPLRRowDFTContext *ctx = reinterpret_cast<ArmPLRRowDFTContext*>(context);
        if (ctx->plan_fwd)   fftwf_destroy_plan(ctx->plan_fwd);
        if (ctx->plan_inv)   fftwf_destroy_plan(ctx->plan_inv);
        if (ctx->fftw_in_r)  fftwf_free(ctx->fftw_in_r);
        if (ctx->fftw_out_c) fftwf_free(ctx->fftw_out_c);
        if (ctx->fftw_in_c)  fftwf_free(ctx->fftw_in_c);
        if (ctx->fftw_out_r) fftwf_free(ctx->fftw_out_r);
        delete ctx;
    }

    return CV_HAL_ERROR_OK;
}

#endif // HAVE_ARMPL
