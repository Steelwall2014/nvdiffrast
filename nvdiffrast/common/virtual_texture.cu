#include "common.h"
#include "virtual_texture.h"

// half is not supported for the original implementation of caAtomicAddTexture
#undef CA_TEMP
#undef CA_TEMP_PARAM
#undef CA_DECLARE_TEMP
#undef CA_SET_GROUP_MASK
#undef CA_SET_GROUP
#undef caAtomicAdd
#undef caAtomicAdd3_xyw
#undef caAtomicAddTexture
#define CA_TEMP _ca_temp
#define CA_TEMP_PARAM float CA_TEMP
#define CA_DECLARE_TEMP(threads_per_block) CA_TEMP_PARAM
#define CA_SET_GROUP_MASK(group, thread_mask)
#define CA_SET_GROUP(group)
#define caAtomicAdd(ptr, value) atomicAdd((ptr), (value))
#define caAtomicAdd3_xyw(ptr, x, y, w)  \
    do {                                \
        atomicAdd((ptr), (x));          \
        atomicAdd((ptr)+1, (y));        \
        atomicAdd((ptr)+3, (w));        \
    } while(0)
#define caAtomicAddTexture(ptr, level, idx, value) atomicAdd((ptr)+(idx), (value))

//------------------------------------------------------------------------
// Memory access and math helpers.

static __device__ __forceinline__ void accum_from_mem(float* a, int s, float  b, float c) { a[0] += b * c; }
static __device__ __forceinline__ void accum_from_mem(float* a, int s, float2 b, float c) { a[0] += b.x * c; a[s] += b.y * c; }
static __device__ __forceinline__ void accum_from_mem(float* a, int s, float4 b, float c) { a[0] += b.x * c; a[s] += b.y * c; a[2*s] += b.z * c; a[3*s] += b.w * c; }
static __device__ __forceinline__ void accum_to_mem(float&  a, float* b, int s) { a += b[0]; }
static __device__ __forceinline__ void accum_to_mem(float2& a, float* b, int s) { float2 v = a; v.x += b[0]; v.y += b[s]; a = v; }
static __device__ __forceinline__ void accum_to_mem(float4& a, float* b, int s) { float4 v = a; v.x += b[0]; v.y += b[s]; v.z += b[2*s]; v.w += b[3*s]; a = v; }
static __device__ __forceinline__ void accum_to_mem(half&  a, float* b, int s) { a += cast<half>(b[0]); }
static __device__ __forceinline__ void accum_to_mem(half2& a, float* b, int s) { half2 v = a; v.x += cast<half>(b[0]); v.y += cast<half>(b[s]); a = v; }
static __device__ __forceinline__ void accum_to_mem(half4& a, float* b, int s) { half4 v = a; v.x += cast<half>(b[0]); v.y += cast<half>(b[s]); v.z += cast<half>(b[2*s]); v.w += cast<half>(b[3*s]); a = v; }
static __device__ __forceinline__ bool isfinite_vec3(const float3& a) { return isfinite(a.x) && isfinite(a.y) && isfinite(a.z); }
static __device__ __forceinline__ bool isfinite_vec4(const float4& a) { return isfinite(a.x) && isfinite(a.y) && isfinite(a.z) && isfinite(a.w); }
template<class T> static __device__ __forceinline__ T lerp  (const T& a, const T& b, float c) { return a + c * (b - a); }
template<> __device__ __forceinline__ half lerp (const half& a, const half& b, float c) { return a + __float2half(c) * (b - a); }
template<> __device__ __forceinline__ half2 lerp (const half2& a, const half2& b, float c) { return half2(lerp(a.x, b.x, c), lerp(a.y, b.y, c)); }
template<> __device__ __forceinline__ half4 lerp (const half4& a, const half4& b, float c) { return half4(lerp(a.x, b.x, c), lerp(a.y, b.y, c), lerp(a.z, b.z, c), lerp(a.w, b.w, c)); }
template<class T> static __device__ __forceinline__ T bilerp(const T& a, const T& b, const T& c, const T& d, const float2& e) { return lerp(lerp(a, b, e.x), lerp(c, d, e.x), e.y); }

//------------------------------------------------------------------------
// General virtual texture indexing.

/*template <bool CUBE_MODE>*/
static __device__ __forceinline__ int2 indexTextureNearest_vt(const VirtualTextureKernelParams& p, float3 uv/*, int tz*/)
{
    int w = p.texWidth;
    int h = p.texHeight;
    float u = uv.x;
    float v = uv.y;

    // Cube map indexing.
    /********TEMPORARILY DISABLE CUBE MODE***********
    if (CUBE_MODE)
    {
        // No wrap. Fold face index into tz right away.
        int idx = indexCubeMap(u, v, uv.z); // Rewrites u, v.
        if (idx < 0)
            return -1; // Invalid uv.
        tz = 6 * tz + idx;
    }
    else
    *********TEMPORARILY DISABLE CUBE MODE**********/
    {
        // Handle boundary.
        if (p.boundaryMode == TEX_BOUNDARY_MODE_WRAP)
        {
            u = u - (float)__float2int_rd(u);
            v = v - (float)__float2int_rd(v);
        }
    }

    u = u * (float)w;
    v = v * (float)h;

    int iu = __float2int_rd(u);
    int iv = __float2int_rd(v);

    // In zero boundary mode, return texture address -1.
    if (/*!CUBE_MODE && */p.boundaryMode == TEX_BOUNDARY_MODE_ZERO)
    {
        if (iu < 0 || iu >= w || iv < 0 || iv >= h)
            return make_int2(0, -1);
    }

    // Otherwise clamp and calculate the coordinate properly.
    iu = min(max(iu, 0), w-1);
    iv = min(max(iv, 0), h-1);

    // Because sometimes the width and height may be smaller than the page size,
    // so we need to decrease the page_size_x and page_size_y to width and height
    int page_size_x = w >= p.page_size_x ? p.page_size_x : w;
    int page_size_y = h >= p.page_size_y ? p.page_size_y : h;
    int pix = iu / page_size_x;     // page index x
    int piy = iv / page_size_y;     // page index y

    iu = iu % page_size_x;
    iv = iv % page_size_y;

    int page_num_x = w / page_size_x; // 2 < p.page_size_x ? 1 : w / p.page_size_x;
    int tc = iu + page_size_x * (iv/* + tz * p.page_size_y*/);
    int indexPage = pix + piy * page_num_x;
    return make_int2(indexPage, tc);
}

/*template <bool CUBE_MODE>*/
static __device__ __forceinline__ float2 indexTextureLinear_vt(const VirtualTextureKernelParams& p, float3 uv/*, int tz*/, int4& indexPageOut, int4& tcOut, int level)
{
    // Mip level size.
    int2 sz = mipLevelSize(p, level);
    int w = sz.x;
    int h = sz.y;

    // Compute texture-space u, v.
    float u = uv.x;
    float v = uv.y;
    bool clampU = false;
    bool clampV = false;

    // Cube map indexing.
    int face = 0;
    /********TEMPORARILY DISABLE CUBE MODE***********
    if (CUBE_MODE)
    {
        // Neither clamp or wrap.
        face = indexCubeMap(u, v, uv.z); // Rewrites u, v.
        if (face < 0)
        {
            tcOut.x = tcOut.y = tcOut.z = tcOut.w = -1; // Invalid uv.
            return make_float2(0.f, 0.f);
        }
        u = u * (float)w - 0.5f;
        v = v * (float)h - 0.5f;
    }
    else
    *********TEMPORARILY DISABLE CUBE MODE**********/
    {
        if (p.boundaryMode == TEX_BOUNDARY_MODE_WRAP)
        {
            // Wrap.
            u = u - (float)__float2int_rd(u);
            v = v - (float)__float2int_rd(v);
        }

        // Move to texel space.
        u = u * (float)w - 0.5f;
        v = v * (float)h - 0.5f;

        if (p.boundaryMode == TEX_BOUNDARY_MODE_CLAMP)
        {
            // Clamp to center of edge texels.
            u = fminf(fmaxf(u, 0.f), w - 1.f);
            v = fminf(fmaxf(v, 0.f), h - 1.f);
            clampU = (u == 0.f || u == w - 1.f);
            clampV = (v == 0.f || v == h - 1.f);
        }
    }

    // Compute texel coordinates and weights.
    int iu0 = __float2int_rd(u);
    int iv0 = __float2int_rd(v);
    int iu1 = iu0 + (clampU ? 0 : 1); // Ensure zero u/v gradients with clamped.
    int iv1 = iv0 + (clampV ? 0 : 1);
    u -= (float)iu0;
    v -= (float)iv0;

    // Cube map wrapping.
    /********TEMPORARILY DISABLE CUBE MODE***********
    bool cubeWrap = CUBE_MODE && (iu0 < 0 || iv0 < 0 || iu1 >= w || iv1 >= h);
    if (cubeWrap)
    {
        tcOut = wrapCubeMap(face, iu0, iu1, iv0, iv1, w);
        tcOut += 6 * tz * w * h;  // Bring in tz.
        return make_float2(u, v); // Done.
    }

    // Fold cube map face into tz.
    if (CUBE_MODE)
        tz = 6 * tz + face;
    *********TEMPORARILY DISABLE CUBE MODE**********/

    // Wrap overflowing texel indices.
    if (/*!CUBE_MODE && */p.boundaryMode == TEX_BOUNDARY_MODE_WRAP)
    {
        if (iu0 < 0) iu0 += w;
        if (iv0 < 0) iv0 += h;
        if (iu1 >= w) iu1 -= w;
        if (iv1 >= h) iv1 -= h;
    }

    // Invalidate texture addresses outside unit square if we are in zero mode.
    if (/*!CUBE_MODE && */p.boundaryMode == TEX_BOUNDARY_MODE_ZERO)
    {
        bool iu0_out = (iu0 < 0 || iu0 >= w);
        bool iu1_out = (iu1 < 0 || iu1 >= w);
        bool iv0_out = (iv0 < 0 || iv0 >= h);
        bool iv1_out = (iv1 < 0 || iv1 >= h);
        if (iu0_out || iv0_out) tcOut.x = -1;
        if (iu1_out || iv0_out) tcOut.y = -1;
        if (iu0_out || iv1_out) tcOut.z = -1;
        if (iu1_out || iv1_out) tcOut.w = -1;

        return make_float2(u, v);   // Exit.
    }

    // Because sometimes the width and height may be smaller than the page size,
    // so we need to decrease the page_size_x and page_size_y to width and height
    int page_size_x = w >= p.page_size_x ? p.page_size_x : w;
    int page_size_y = h >= p.page_size_y ? p.page_size_y : h;
    int page_num_x = w / page_size_x;

    int ipx0 = iu0 / page_size_x;
    int ipx1 = iu1 / page_size_x;
    int ipy0 = iv0 / page_size_y;
    int ipy1 = iv1 / page_size_y;

    iu0 = iu0 % page_size_x;
    iu1 = iu1 % page_size_x;
    iv0 = iv0 % page_size_y;
    iv1 = iv1 % page_size_y;

    // Coordinates with tz folded in.
    int iu0z = iu0/* + tz * p.page_size_x * p.page_size_y*/;
    int iu1z = iu1/* + tz * p.page_size_x * p.page_size_y*/;
    tcOut.x = iu0z + page_size_x * iv0;
    tcOut.y = iu1z + page_size_x * iv0;
    tcOut.z = iu0z + page_size_x * iv1;
    tcOut.w = iu1z + page_size_x * iv1;

    indexPageOut.x = ipx0 + ipy0 * page_num_x;
    indexPageOut.y = ipx1 + ipy0 * page_num_x;
    indexPageOut.z = ipx0 + ipy1 * page_num_x;
    indexPageOut.w = ipx1 + ipy1 * page_num_x;

    // All done.
    return make_float2(u, v);
}

//------------------------------------------------------------------------
// Mip level calculation.

template </*bool CUBE_MODE, */bool BIAS_ONLY, int FILTER_MODE>
static __device__ __forceinline__ void calculateMipLevel_vt(int& level0, int& level1, float& flevel, const VirtualTextureKernelParams& p, int pidx, float3 uv, float4* pdw, float3* pdfdv)
{
    // Do nothing if mips not in use.
    if (FILTER_MODE == TEX_MODE_NEAREST || FILTER_MODE == TEX_MODE_LINEAR)
        return;

    // Determine mip level based on UV pixel derivatives. If no derivatives are given (mip level bias only), leave as zero.
    if (!BIAS_ONLY)
    {
        // Get pixel derivatives of texture coordinates.
        float4 uvDA;
        float3 dvdX, dvdY; // Gradients use these later.
        /********TEMPORARILY DISABLE CUBE MODE***********
        if (CUBE_MODE)
        {
            // Fetch.
            float2 d0 = ((const float2*)p.uvDA)[3 * pidx + 0];
            float2 d1 = ((const float2*)p.uvDA)[3 * pidx + 1];
            float2 d2 = ((const float2*)p.uvDA)[3 * pidx + 2];

            // Map d{x,y,z}/d{X,Y} into d{s,t}/d{X,Y}.
            dvdX = make_float3(d0.x, d1.x, d2.x); // d{x,y,z}/dX
            dvdY = make_float3(d0.y, d1.y, d2.y); // d{x,y,z}/dY
            uvDA = indexCubeMapGradST(uv, dvdX, dvdY); // d{s,t}/d{X,Y}
        }
        else
        *********TEMPORARILY DISABLE CUBE MODE**********/
        {
            // Fetch.
            uvDA = ((const float4*)p.uvDA)[pidx];
        }

        // Scaling factors.
        float uscl = p.texWidth;
        float vscl = p.texHeight;

        // d[s,t]/d[X,Y].
        float dsdx = uvDA.x * uscl;
        float dsdy = uvDA.y * uscl;
        float dtdx = uvDA.z * vscl;
        float dtdy = uvDA.w * vscl;

        // Calculate footprint axis lengths.
        float A = dsdx*dsdx + dtdx*dtdx;
        float B = dsdy*dsdy + dtdy*dtdy;
        float C = dsdx*dsdy + dtdx*dtdy;
        float l2b = 0.5 * (A + B);
        float l2n = 0.25 * (A-B)*(A-B) + C*C;
        float l2a = sqrt(l2n);
        float lenMinorSqr = fmaxf(0.0, l2b - l2a);
        float lenMajorSqr = l2b + l2a;

        // Footprint vs. mip level gradient.
        if (pdw && FILTER_MODE == TEX_MODE_LINEAR_MIPMAP_LINEAR)
        {
            float dw   = 0.72134752f / (l2n + l2a * l2b); // Constant is 0.5/ln(2).
            float AB   = dw * .5f * (A - B);
            float Cw   = dw * C;
            float l2aw = dw * l2a;
            float d_f_ddsdX = uscl * (dsdx * (l2aw + AB) + dsdy * Cw);
            float d_f_ddsdY = uscl * (dsdy * (l2aw - AB) + dsdx * Cw);
            float d_f_ddtdX = vscl * (dtdx * (l2aw + AB) + dtdy * Cw);
            float d_f_ddtdY = vscl * (dtdy * (l2aw - AB) + dtdx * Cw);

            float4 d_f_dw = make_float4(d_f_ddsdX, d_f_ddsdY, d_f_ddtdX, d_f_ddtdY);
            /*if (!CUBE_MODE)*/
                *pdw = isfinite_vec4(d_f_dw) ? d_f_dw : make_float4(0.f, 0.f, 0.f, 0.f);

            // In cube maps, there is also a texture coordinate vs. mip level gradient.
            // Only output nonzero vectors if both are free of inf/Nan garbage.
            /********TEMPORARILY DISABLE CUBE MODE***********
            if (CUBE_MODE)
            {
                float4 dx, dy, dz;
                indexCubeMapGrad2(uv, dvdX, dvdY, dx, dy, dz);
                float3 d_dsdX_dv = make_float3(dx.x, dy.x, dz.x);
                float3 d_dsdY_dv = make_float3(dx.y, dy.y, dz.y);
                float3 d_dtdX_dv = make_float3(dx.z, dy.z, dz.z);
                float3 d_dtdY_dv = make_float3(dx.w, dy.w, dz.w);

                float3 d_f_dv = make_float3(0.f, 0.f, 0.f);
                d_f_dv += d_dsdX_dv * d_f_ddsdX;
                d_f_dv += d_dsdY_dv * d_f_ddsdY;
                d_f_dv += d_dtdX_dv * d_f_ddtdX;
                d_f_dv += d_dtdY_dv * d_f_ddtdY;

                bool finite = isfinite_vec4(d_f_dw) && isfinite_vec3(d_f_dv);
                *pdw   = finite ? d_f_dw : make_float4(0.f, 0.f, 0.f, 0.f);
                *pdfdv = finite ? d_f_dv : make_float3(0.f, 0.f, 0.f);
            }
            *********TEMPORARILY DISABLE CUBE MODE**********/
        }

        // Finally, calculate mip level.
        flevel = .5f * __log2f(lenMajorSqr); // May be inf/NaN, but clamp fixes it.
    }

    // Bias the mip level and clamp.
    if (p.mipLevelBias)
        flevel += p.mipLevelBias[pidx];
    flevel = fminf(fmaxf(flevel, 0.f), (float)p.mipLevelMax);

    // Calculate levels depending on filter mode.
    level0 = __float2int_rd(flevel);

    // Leave everything else at zero if flevel == 0 (magnification) or when in linear-mipmap-nearest mode.
    if (FILTER_MODE == TEX_MODE_LINEAR_MIPMAP_LINEAR && flevel > 0.f)
    {
        level1 = min(level0 + 1, p.mipLevelMax);
        flevel -= level0; // Fractional part. Zero if clamped on last level.
    }
}

//------------------------------------------------------------------------
// Texel fetch and accumulator helpers that understand cube map corners.

template<class T, typename value_type=typename value_type_traits<T>::type>
static __device__ __forceinline__ void fetchQuad_vt(T& a00, T& a10, T& a01, T& a11, const value_type** pIn, int4 pi, int4 tc/*, bool corner*/)
{
    // For invalid cube map uv, tc will be all negative, and all texel values will be zero.
    /*if (corner)
    {
        T avg = zero_value<T>();
        if (tc.x >= 0) avg += (a00 = *((const T*)&pIn[pi.x][tc.x]));
        if (tc.y >= 0) avg += (a10 = *((const T*)&pIn[pi.y][tc.y]));
        if (tc.z >= 0) avg += (a01 = *((const T*)&pIn[pi.z][tc.z]));
        if (tc.w >= 0) avg += (a11 = *((const T*)&pIn[pi.w][tc.w]));
        avg *= 0.33333333f;
        if (tc.x < 0) a00 = avg;
        if (tc.y < 0) a10 = avg;
        if (tc.z < 0) a01 = avg;
        if (tc.w < 0) a11 = avg;
    }
    else*/
    {
        a00 = (tc.x >= 0) ? *((const T*)&pIn[pi.x][tc.x]) : zero_value<T>();
        a10 = (tc.y >= 0) ? *((const T*)&pIn[pi.y][tc.y]) : zero_value<T>();
        a01 = (tc.z >= 0) ? *((const T*)&pIn[pi.z][tc.z]) : zero_value<T>();
        a11 = (tc.w >= 0) ? *((const T*)&pIn[pi.w][tc.w]) : zero_value<T>();
    }
}

template<class T>
static __device__ __forceinline__ void accumQuad_vt(float4 c, T** pOut, int level, int4 pi, int4 tc/*, bool corner*/, CA_TEMP_PARAM)
{
    // For invalid cube map uv, tc will be all negative, and no accumulation will take place.
    /*if (corner)
    {
        float cb;
        if (tc.x < 0) cb = c.x;
        if (tc.y < 0) cb = c.y;
        if (tc.z < 0) cb = c.z;
        if (tc.w < 0) cb = c.w;
        cb *= 0.33333333f;
        if (tc.x >= 0) caAtomicAddTexture(pOut[pi.x], level, tc.x, c.x + cb);
        if (tc.y >= 0) caAtomicAddTexture(pOut[pi.y], level, tc.y, c.y + cb);
        if (tc.z >= 0) caAtomicAddTexture(pOut[pi.z], level, tc.z, c.z + cb);
        if (tc.w >= 0) caAtomicAddTexture(pOut[pi.w], level, tc.w, c.w + cb);
    }
    else*/
    {
        if (tc.x >= 0) caAtomicAddTexture(pOut[pi.x], level, tc.x, cast<T>(c.x));
        if (tc.y >= 0) caAtomicAddTexture(pOut[pi.y], level, tc.y, cast<T>(c.y));
        if (tc.z >= 0) caAtomicAddTexture(pOut[pi.z], level, tc.z, cast<T>(c.z));
        if (tc.w >= 0) caAtomicAddTexture(pOut[pi.w], level, tc.w, cast<T>(c.w));
    }
}

//------------------------------------------------------------------------
// Mip level calculation.

/*template <bool CUBE_MODE>*/
static __device__ __forceinline__ int4 calcPixelCoverage(const VirtualTextureKernelParams& p, float3 uv, int level)
{
    // Mip level size.
    int2 sz = mipLevelSize(p, level);
    int w = sz.x;
    int h = sz.y;
    int4 coverage;

    // Compute texture-space u, v.
    float u = uv.x;
    float v = uv.y;
    bool clampU = false;
    bool clampV = false;

    if (p.boundaryMode == TEX_BOUNDARY_MODE_WRAP)
    {
        // Wrap.
        u = u - (float)__float2int_rd(u);
        v = v - (float)__float2int_rd(v);
    }

    // Move to texel space.
    u = u * (float)w - 0.5f;
    v = v * (float)h - 0.5f;

    if (p.boundaryMode == TEX_BOUNDARY_MODE_CLAMP)
    {
        // Clamp to center of edge texels.
        u = fminf(fmaxf(u, 0.f), w - 1.f);
        v = fminf(fmaxf(v, 0.f), h - 1.f);
        clampU = (u == 0.f || u == w - 1.f);
        clampV = (v == 0.f || v == h - 1.f);
    }

    // Compute texel coordinates and weights.
    int iu0 = __float2int_rd(u);
    int iv0 = __float2int_rd(v);
    int iu1 = iu0 + (clampU ? 0 : 1); // Ensure zero u/v gradients with clamped.
    int iv1 = iv0 + (clampV ? 0 : 1);
    u -= (float)iu0;
    v -= (float)iv0;

    // Wrap overflowing texel indices.
    if (p.boundaryMode == TEX_BOUNDARY_MODE_WRAP)
    {
        if (iu0 < 0) iu0 += w;
        if (iv0 < 0) iv0 += h;
        if (iu1 >= w) iu1 -= w;
        if (iv1 >= h) iv1 -= h;
    }

    // Invalidate texture addresses outside unit square if we are in zero mode.
    if (p.boundaryMode == TEX_BOUNDARY_MODE_ZERO)
    {
        bool iu0_out = (iu0 < 0 || iu0 >= w);
        bool iu1_out = (iu1 < 0 || iu1 >= w);
        bool iv0_out = (iv0 < 0 || iv0 >= h);
        bool iv1_out = (iv1 < 0 || iv1 >= h);
        if (iu0_out || iv0_out) coverage.x = -1;
        if (iu1_out || iv0_out) coverage.y = -1;
        if (iu0_out || iv1_out) coverage.z = -1;
        if (iu1_out || iv1_out) coverage.w = -1;

        return coverage;   // Exit.
    }

    int2 sz0 = mipLevelSize(p, 0);
    int2 sz1 = mipLevelSize(p, level);
    int scale_x = sz0.x / sz1.x;
    int scale_y = sz0.y / sz1.y;
    int start_x = iu0 * scale_x;
    int end_x = (iu1+1) * scale_x - 1;
    int start_y = iv0 * scale_y;
    int end_y = (iv1+1) * scale_y - 1;

    int page_size_x = p.texWidth >= p.page_size_x ? p.page_size_x : p.texWidth;
    int page_size_y = p.texHeight >= p.page_size_y ? p.page_size_y : p.texHeight;
    int start_pi_x = start_x / page_size_x;
    int end_pi_x = end_x / page_size_x;
    int start_pi_y = start_y / page_size_y;
    int end_pi_y = end_y / page_size_y;

    coverage.x = start_pi_x;
    coverage.y = end_pi_x;
    coverage.z = start_pi_y;
    coverage.w = end_pi_y;

    // All done.
    return coverage;
}

static __forceinline__ __device__ int calcPageIndex(const VirtualTextureKernelParams& p, int pix, int piy)
{
    int page_size_x = p.texWidth >= p.page_size_x ? p.page_size_x : p.texWidth;
    int page_size_y = p.texHeight >= p.page_size_y ? p.page_size_y : p.texHeight;

    int page_num_x = p.texWidth / page_size_x;

    return pix + piy * page_num_x;
}

template <int C, bool BIAS_ONLY, int FILTER_MODE>
static __forceinline__ __device__ void VirtualTextureFeedbackKernelTemplate(const VirtualTextureKernelParams p)
{
    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.imgWidth || py >= p.imgHeight || pz >= p.n)
        return;

    // Pixel index.
    int pidx = px + p.imgWidth * (py + p.imgHeight * pz);

    if (p.mask && !p.mask[pidx])
        return;

    // Get UV.
    float3 uv = make_float3(((const float2*)p.uv)[pidx], 0.f);

    // Nearest mode.
    if (FILTER_MODE == TEX_MODE_NEAREST)
    {
        int2 pi_tc = indexTextureNearest_vt(p, uv);
        bool* pOut = p.feedback[0];
        pOut[pi_tc.x] = true;

        return; // Exit.
    }
    
    // Calculate mip level. In 'linear' mode these will all stay zero.
    float  flevel = 0.f; // Fractional level.
    int    level0 = 0;   // Discrete level 0.
    int    level1 = 0;   // Discrete level 1.
    calculateMipLevel_vt<BIAS_ONLY, FILTER_MODE>(level0, level1, flevel, p, pidx, uv, 0, 0);

    // Mark the texels that will be directly accessed
    {
        // Get texel indices and pointer for level 0.
        int4 tc0 = make_int4(0, 0, 0, 0);
        int4 pi0 = make_int4(0, 0, 0, 0);    // page index
        float2 uv0 = indexTextureLinear_vt(p, uv, pi0, tc0, level0);
        bool* pOut0 = p.feedback[level0];

        pOut0[pi0.x] = true;
        pOut0[pi0.y] = true;
        pOut0[pi0.z] = true;
        pOut0[pi0.w] = true;

        if (FILTER_MODE == TEX_MODE_LINEAR || FILTER_MODE == TEX_MODE_LINEAR_MIPMAP_NEAREST)
        {
            return; // Exit.
        }

        // Get texel indices and pointer for level 1.
        int4 tc1 = make_int4(0, 0, 0, 0);
        int4 pi1 = make_int4(0, 0, 0, 0);    // page index
        float2 uv1 = indexTextureLinear_vt(p, uv, pi1, tc1, level1);
        bool* pOut1 = p.feedback[level1];

        if (flevel > 0.f)
        {
            pOut1[pi1.x] = true;
            pOut1[pi1.y] = true;
            pOut1[pi1.z] = true;
            pOut1[pi1.w] = true;
        }
    }

    // A texel of higher mipmap level covers more than one texels of the first mipmap level.
    // So some texels of the first mipmap level are not directly accessed, but their 
    // gradients may be affected by the texels of higher mipmap levels.
    // That's why we need the following codes to calculate the coverage.
    {
        bool* pOut = p.feedback[0];
        int page_num_x = calcPageNum(p.texWidth, p.page_size_x);
        int page_num_y = calcPageNum(p.texHeight, p.page_size_y);
        int4 coverage0 = calcPixelCoverage(p, uv, level0);
        int start_pi_x0 = coverage0.x;
        int end_pi_x0 = coverage0.x <= coverage0.y ? coverage0.y : coverage0.y+page_num_x;
        int start_pi_y0 = coverage0.z;
        int end_pi_y0 = coverage0.z <= coverage0.w ? coverage0.w : coverage0.w+page_num_y;
        for (int pi_y = start_pi_y0; pi_y <= end_pi_y0; pi_y++)
        {
            for (int pi_x = start_pi_x0; pi_x <= end_pi_x0; pi_x++)
            {
                int pi = calcPageIndex(p, pi_x%page_num_x, pi_y%page_num_y);
                pOut[pi] = true;
            }
        }

        if (FILTER_MODE == TEX_MODE_LINEAR || FILTER_MODE == TEX_MODE_LINEAR_MIPMAP_NEAREST)
        {
            return; // Exit.
        }

        int4 coverage1 = calcPixelCoverage(p, uv, level1);
        int start_pi_x1 = coverage1.x;
        int end_pi_x1 = coverage1.x <= coverage1.y ? coverage1.y : coverage1.y+page_num_x;
        int start_pi_y1 = coverage1.z;
        int end_pi_y1 = coverage1.z <= coverage1.w ? coverage1.w : coverage1.w+page_num_y;
        if (flevel > 0.f)
        {
            for (int pi_y = start_pi_y1; pi_y <= end_pi_y1; pi_y++)
            {
                for (int pi_x = start_pi_x1; pi_x <= end_pi_x1; pi_x++)
                {
                    int pi = calcPageIndex(p, pi_x%page_num_x, pi_y%page_num_y);
                    pOut[pi] = true;
                }
            }
        }
    
    }
}

// Template specializations.
__global__ void VirtualTextureFeedbackKernelNearest1                    (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<1, false, TEX_MODE_NEAREST>(p); }
__global__ void VirtualTextureFeedbackKernelNearest2                    (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<2, false, TEX_MODE_NEAREST>(p); }
__global__ void VirtualTextureFeedbackKernelNearest4                    (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<4, false, TEX_MODE_NEAREST>(p); }
__global__ void VirtualTextureFeedbackKernelLinear1                     (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<1, false, TEX_MODE_LINEAR>(p); }
__global__ void VirtualTextureFeedbackKernelLinear2                     (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<2, false, TEX_MODE_LINEAR>(p); }
__global__ void VirtualTextureFeedbackKernelLinear4                     (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<4, false, TEX_MODE_LINEAR>(p); }
__global__ void VirtualTextureFeedbackKernelLinearMipmapNearest1        (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<1, false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFeedbackKernelLinearMipmapNearest2        (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<2, false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFeedbackKernelLinearMipmapNearest4        (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<4, false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFeedbackKernelLinearMipmapLinear1         (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<1, false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureFeedbackKernelLinearMipmapLinear2         (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<2, false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureFeedbackKernelLinearMipmapLinear4         (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<4, false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureFeedbackKernelLinearMipmapNearestBO1      (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<1, true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFeedbackKernelLinearMipmapNearestBO2      (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<2, true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFeedbackKernelLinearMipmapNearestBO4      (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<4, true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFeedbackKernelLinearMipmapLinearBO1       (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<1, true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureFeedbackKernelLinearMipmapLinearBO2       (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<2, true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureFeedbackKernelLinearMipmapLinearBO4       (const VirtualTextureKernelParams p) { VirtualTextureFeedbackKernelTemplate<4, true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }

//------------------------------------------------------------------------
// Virtual texture forward kernel

template <class T, int C, bool BIAS_ONLY, int FILTER_MODE>
static __forceinline__ __device__ void VirtualTextureFwdKernelTemplate(const VirtualTextureKernelParams p)
{
    using value_type = typename value_type_traits<T>::type;
    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    int tz = (p.texDepth == 1) ? 0 : pz;
    if (px >= p.imgWidth || py >= p.imgHeight || pz >= p.n)
        return;

    // Pixel index.
    int pidx = px + p.imgWidth * (py + p.imgHeight * pz);

    if (p.mask && !p.mask[pidx])
        return;

    // Output ptr.
    value_type* pOut = (value_type*)p.out + pidx * p.channels;

    // Get UV.
    float3 uv = make_float3(((const float2*)p.uv)[pidx], 0.f);

    // Nearest mode.
    if (FILTER_MODE == TEX_MODE_NEAREST)
    {
        int2 pi_tc = indexTextureNearest_vt(p, uv);
        int page_index = pi_tc.x;
        int tc = pi_tc.y;
        tc *= p.channels;
        const value_type* pIn = (const value_type*)p.tex[0][page_index];

        // Copy if valid tc, otherwise output zero.
        for (int i=0; i < p.channels; i += C)
        {
            *((T*)&pOut[i]) = (tc >= 0) ? *((const T*)&pIn[tc + i]) : zero_value<T>();
        }
        

        return; // Exit.
    }

    // Calculate mip level. In 'linear' mode these will all stay zero.
    float  flevel = 0.f; // Fractional level.
    int    level0 = 0;   // Discrete level 0.
    int    level1 = 0;   // Discrete level 1.
    calculateMipLevel_vt<BIAS_ONLY, FILTER_MODE>(level0, level1, flevel, p, pidx, uv, 0, 0);

    // Get texel indices and pointer for level 0.
    int4 tc0 = make_int4(0, 0, 0, 0);
    int4 pi0 = make_int4(0, 0, 0, 0);
    float2 uv0 = indexTextureLinear_vt(p, uv, pi0, tc0, level0);
    const value_type** pIn0 = (const value_type**)p.tex[level0];
    // bool corner0 = CUBE_MODE && ((tc0.x | tc0.y | tc0.z | tc0.w) < 0);
    tc0 *= p.channels;

    // Bilinear fetch.
    if (FILTER_MODE == TEX_MODE_LINEAR || FILTER_MODE == TEX_MODE_LINEAR_MIPMAP_NEAREST)
    {
        // Interpolate.
        for (int i=0; i < p.channels; i += C, tc0 += C)
        {
            T a00, a10, a01, a11;
            fetchQuad_vt(a00, a10, a01, a11, pIn0, pi0, tc0);
            *((T*)&pOut[i]) = bilerp(a00, a10, a01, a11, uv0);
        }
        return; // Exit.
    }

    // Get texel indices and pointer for level 1.
    int4 tc1 = make_int4(0, 0, 0, 0);
    int4 pi1 = make_int4(0, 0, 0, 0);
    float2 uv1 = indexTextureLinear_vt(p, uv, pi1, tc1, level1);
    const value_type** pIn1 = (const value_type**)p.tex[level1];
    // bool corner1 = CUBE_MODE && ((tc1.x | tc1.y | tc1.z | tc1.w) < 0);
    tc1 *= p.channels;

    // Trilinear fetch.
    for (int i=0; i < p.channels; i += C, tc0 += C, tc1 += C)
    {
        // First level.
        T a00, a10, a01, a11;
        fetchQuad_vt(a00, a10, a01, a11, pIn0, pi0, tc0);
        T a = bilerp(a00, a10, a01, a11, uv0);

        // Second level unless in magnification mode.
        if (flevel > 0.f)
        {
            T b00, b10, b01, b11;
            fetchQuad_vt(b00, b10, b01, b11, pIn1, pi1, tc1);
            T b = bilerp(b00, b10, b01, b11, uv1);
            a = lerp(a, b, flevel); // Interpolate between levels.
        }

        // Write.
        *((T*)&pOut[i]) = a;
    }
}

// Template specializations.
__global__ void VirtualTextureFwdKernelNearest1                    (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float,  1, false, TEX_MODE_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelNearest2                    (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float2, 2, false, TEX_MODE_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelNearest4                    (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float4, 4, false, TEX_MODE_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelLinear1                     (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float,  1, false, TEX_MODE_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelLinear2                     (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float2, 2, false, TEX_MODE_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelLinear4                     (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float4, 4, false, TEX_MODE_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapNearest1        (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float,  1, false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapNearest2        (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float2, 2, false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapNearest4        (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float4, 4, false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapLinear1         (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float,  1, false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapLinear2         (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float2, 2, false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapLinear4         (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float4, 4, false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapNearestBO1      (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float,  1, true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapNearestBO2      (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float2, 2, true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapNearestBO4      (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float4, 4, true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapLinearBO1       (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float,  1, true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapLinearBO2       (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float2, 2, true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapLinearBO4       (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<float4, 4, true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelNearestHalf1                (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half,  1, false, TEX_MODE_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelNearestHalf2                (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half2, 2, false, TEX_MODE_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelNearestHalf4                (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half4, 4, false, TEX_MODE_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelLinearHalf1                 (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half,  1, false, TEX_MODE_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelLinearHalf2                 (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half2, 2, false, TEX_MODE_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelLinearHalf4                 (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half4, 4, false, TEX_MODE_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapNearestHalf1    (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half,  1, false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapNearestHalf2    (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half2, 2, false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapNearestHalf4    (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half4, 4, false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapLinearHalf1     (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half,  1, false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapLinearHalf2     (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half2, 2, false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapLinearHalf4     (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half4, 4, false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapNearestBOHalf1  (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half,  1, true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapNearestBOHalf2  (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half2, 2, true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapNearestBOHalf4  (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half4, 4, true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapLinearBOHalf1   (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half,  1, true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapLinearBOHalf2   (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half2, 2, true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureFwdKernelLinearMipmapLinearBOHalf4   (const VirtualTextureKernelParams p) { VirtualTextureFwdKernelTemplate<half4, 4, true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }

//------------------------------------------------------------------------
// Virtual texture gradient kernel

template <typename T, bool BIAS_ONLY, int FILTER_MODE>
static __forceinline__ __device__ void VirtualTextureGradKernelTemplate(const VirtualTextureKernelParams p)
{
    // Temporary space for coalesced atomics.
    CA_DECLARE_TEMP(TEX_GRAD_MAX_KERNEL_BLOCK_WIDTH * TEX_GRAD_MAX_KERNEL_BLOCK_HEIGHT);

    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    int tz = (p.texDepth == 1) ? 0 : pz;
    if (px >= p.imgWidth || py >= p.imgHeight || pz >= p.n)
        return;

    // Pixel index.
    int pidx = px + p.imgWidth * (py + p.imgHeight * pz);

    if (p.mask && !p.mask[pidx])
        return;

    // Early exit if output gradients are zero.
    const T* pDy = (const T*)p.dy + pidx * p.channels;
    unsigned int dmax = 0u;
    for (int i=0; i < p.channels; i++)
        dmax |= __float_as_uint(cast<float>(pDy[i]));

    // Store zeros and exit.
    if (__uint_as_float(dmax) == 0.f)
    {
        /********TEMPORARILY DISABLE CUBE MODE***********
        if (CUBE_MODE)
        {
            if (FILTER_MODE != TEX_MODE_NEAREST)
                ((float3*)p.gradUV)[pidx] = make_float3(0.f, 0.f, 0.f);
            if (FILTER_MODE == TEX_MODE_LINEAR_MIPMAP_LINEAR)
            {
                if (p.gradUVDA)
                {
                    ((float2*)p.gradUVDA)[3 * pidx + 0] = make_float2(0.f, 0.f);
                    ((float2*)p.gradUVDA)[3 * pidx + 1] = make_float2(0.f, 0.f);
                    ((float2*)p.gradUVDA)[3 * pidx + 2] = make_float2(0.f, 0.f);
                }
                if (p.gradMipLevelBias)
                    p.gradMipLevelBias[pidx] = 0.f;
            }
        }
        else
        *********TEMPORARILY DISABLE CUBA MODE**********/
        {
            if (FILTER_MODE != TEX_MODE_NEAREST)
                ((float2*)p.gradUV)[pidx] = make_float2(0.f, 0.f);
            if (FILTER_MODE == TEX_MODE_LINEAR_MIPMAP_LINEAR)
            {
                if (p.gradUVDA)
                    ((float4*)p.gradUVDA)[pidx] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (p.gradMipLevelBias)
                    p.gradMipLevelBias[pidx] = 0.f;
            }
        }
        return;
    }
    
    // Get UV.
    float3 uv = make_float3(((const float2*)p.uv)[pidx], 0.f);
       
    // Nearest mode - texture gradients only.
    if (FILTER_MODE == TEX_MODE_NEAREST)
    {
        int2 pi_tc = indexTextureNearest_vt(p, uv);
        int page_index = pi_tc.x;
        int tc = pi_tc.y;
        if (tc < 0)
            return; // Outside texture.

        tc *= p.channels;
        T* pOut = (T*)p.gradTex[0][page_index];

        // Accumulate texture gradients.
        for (int i=0; i < p.channels; i++)
        {
            caAtomicAddTexture(pOut, 0, tc + i, pDy[i]);
        }

        return; // Exit.
    }

    // Calculate mip level. In 'linear' mode these will all stay zero.
    float4 dw = make_float4(0.f, 0.f, 0.f, 0.f);
    float3 dfdv = make_float3(0.f, 0.f, 0.f);
    float  flevel = 0.f; // Fractional level.
    int    level0 = 0;   // Discrete level 0.
    int    level1 = 0;   // Discrete level 1.
    calculateMipLevel_vt<BIAS_ONLY, FILTER_MODE>(level0, level1, flevel, p, pidx, uv, &dw, &dfdv);

    // UV gradient accumulators.
    float gu = 0.f;
    float gv = 0.f;

    // Get texel indices and pointers for level 0.
    int4 tc0 = make_int4(0, 0, 0, 0);
    int4 pi0 = make_int4(0, 0, 0, 0);
    float2 uv0 = indexTextureLinear_vt(p, uv, pi0, tc0, level0);
    const T** pIn0 = (const T**)p.tex[level0];
    T** pOut0 = (T**)p.gradTex[level0];
    // bool corner0 = CUBE_MODE && ((tc0.x | tc0.y | tc0.z | tc0.w) < 0);
    tc0 *= p.channels;

    // Texel weights.
    float uv011 = uv0.x * uv0.y;
    float uv010 = uv0.x - uv011;
    float uv001 = uv0.y - uv011;
    float uv000 = 1.f - uv0.x - uv001;
    float4 tw0 = make_float4(uv000, uv010, uv001, uv011);

    // Attribute weights.
    int2 sz0 = mipLevelSize(p, level0);
    float sclu0 = (float)sz0.x;
    float sclv0 = (float)sz0.y;

    // Bilinear mode - texture and uv gradients.
    if (FILTER_MODE == TEX_MODE_LINEAR || FILTER_MODE == TEX_MODE_LINEAR_MIPMAP_NEAREST)
    {
        for (int i=0; i < p.channels; i++, tc0 += 1)
        {
            float dy = cast<float>(pDy[i]);
            accumQuad_vt(tw0 * dy, pOut0, level0, pi0, tc0, CA_TEMP);

            float a00, a10, a01, a11;
            {
                T a00_, a10_, a01_, a11_;
                fetchQuad_vt(a00_, a10_, a01_, a11_, pIn0, pi0, tc0);
                a00 = cast<float>(a00_);
                a10 = cast<float>(a10_);
                a01 = cast<float>(a01_);
                a11 = cast<float>(a11_);
            }
            float ad = (a11 + a00 - a10 - a01);
            gu += dy * ((a10 - a00) + uv0.y * ad) * sclu0;
            gv += dy * ((a01 - a00) + uv0.x * ad) * sclv0;
        }

        // Store UV gradients and exit.
        /*if (CUBE_MODE)
            ((float3*)p.gradUV)[pidx] = indexCubeMapGrad(uv, gu, gv);
        else*/
            ((float2*)p.gradUV)[pidx] = make_float2(gu, gv);

        return;
    }

    // Accumulate fractional mip level gradient.
    float df = 0; // dL/df.

    // Get texel indices and pointers for level 1.
    int4 tc1 = make_int4(0, 0, 0, 0);
    int4 pi1 = make_int4(0, 0, 0, 0);
    float2 uv1 = indexTextureLinear_vt(p, uv, pi1, tc1, level1);
    const T** pIn1 = (const T**)p.tex[level1];
    T** pOut1 = (T**)p.gradTex[level1];
    // bool corner1 = CUBE_MODE && ((tc1.x | tc1.y | tc1.z | tc1.w) < 0);
    tc1 *= p.channels;

    // Texel weights.
    float uv111 = uv1.x * uv1.y;
    float uv110 = uv1.x - uv111;
    float uv101 = uv1.y - uv111;
    float uv100 = 1.f - uv1.x - uv101;
    float4 tw1 = make_float4(uv100, uv110, uv101, uv111);

    // Attribute weights.
    int2 sz1 = mipLevelSize(p, level1);
    float sclu1 = (float)sz1.x;
    float sclv1 = (float)sz1.y;

    // Trilinear mode.
    for (int i=0; i < p.channels; i++, tc0 += 1, tc1 += 1)
    {
        float dy = cast<float>(pDy[i]);
        float dy0 = (1.f - flevel) * dy;
        accumQuad_vt(tw0 * dy0, pOut0, level0, pi0, tc0, CA_TEMP);

        // UV gradients for first level.
        float a00, a10, a01, a11;
        {
            T a00_, a10_, a01_, a11_;
            fetchQuad_vt(a00_, a10_, a01_, a11_, pIn0, pi0, tc0);
            a00 = cast<float>(a00_);
            a10 = cast<float>(a10_);
            a01 = cast<float>(a01_);
            a11 = cast<float>(a11_);
        }
        float ad = (a11 + a00 - a10 - a01);
        gu += dy0 * ((a10 - a00) + uv0.y * ad) * sclu0;
        gv += dy0 * ((a01 - a00) + uv0.x * ad) * sclv0;

        // Second level unless in magnification mode.
        if (flevel > 0.f)
        {
            // Texture gradients for second level.
            float dy1 = flevel * dy;
            accumQuad_vt(tw1 * dy1, pOut1, level1, pi1, tc1, CA_TEMP);

            // UV gradients for second level.
            float b00, b10, b01, b11;
            {
                T b00_, b10_, b01_, b11_;
                fetchQuad_vt(b00_, b10_, b01_, b11_, pIn1, pi1, tc1);
                b00 = cast<float>(b00_);
                b10 = cast<float>(b10_);
                b01 = cast<float>(b01_);
                b11 = cast<float>(b11_);
            }
            float bd = (b11 + b00 - b10 - b01);
            gu += dy1 * ((b10 - b00) + uv1.y * bd) * sclu1;
            gv += dy1 * ((b01 - b00) + uv1.x * bd) * sclv1;

            // Mip level gradient.
            float a = bilerp(a00, a10, a01, a11, uv0);
            float b = bilerp(b00, b10, b01, b11, uv1);
            df += (b-a) * dy;
        }
    }

    // Store UV gradients.
    /*if (CUBE_MODE)
        ((float3*)p.gradUV)[pidx] = indexCubeMapGrad(uv, gu, gv) + (dfdv * df);
    else*/
        ((float2*)p.gradUV)[pidx] = make_float2(gu, gv);

    // Store mip level bias gradient.
    if (p.gradMipLevelBias)
        p.gradMipLevelBias[pidx] = df;

    // Store UV pixel differential gradients.
    if (!BIAS_ONLY)
    {
        // Final gradients.
        dw *= df; // dL/(d{s,y}/d{X,Y}) = df/(d{s,y}/d{X,Y}) * dL/df.

        // Store them.
        /*if (CUBE_MODE)
        {
            // Remap from dL/(d{s,t}/s{X,Y}) to dL/(d{x,y,z}/d{X,Y}).
            float3 g0, g1;
            indexCubeMapGrad4(uv, dw, g0, g1);
            ((float2*)p.gradUVDA)[3 * pidx + 0] = make_float2(g0.x, g1.x);
            ((float2*)p.gradUVDA)[3 * pidx + 1] = make_float2(g0.y, g1.y);
            ((float2*)p.gradUVDA)[3 * pidx + 2] = make_float2(g0.z, g1.z);
        }
        else*/
            ((float4*)p.gradUVDA)[pidx] = dw;
    }
}

// Template specializations.
__global__ void VirtualTextureGradKernelNearest                    (const VirtualTextureKernelParams p) { VirtualTextureGradKernelTemplate<float, false, TEX_MODE_NEAREST>(p); }
__global__ void VirtualTextureGradKernelLinear                     (const VirtualTextureKernelParams p) { VirtualTextureGradKernelTemplate<float, false, TEX_MODE_LINEAR>(p); }
__global__ void VirtualTextureGradKernelLinearMipmapNearest        (const VirtualTextureKernelParams p) { VirtualTextureGradKernelTemplate<float, false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureGradKernelLinearMipmapLinear         (const VirtualTextureKernelParams p) { VirtualTextureGradKernelTemplate<float, false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureGradKernelLinearMipmapNearestBO      (const VirtualTextureKernelParams p) { VirtualTextureGradKernelTemplate<float, true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureGradKernelLinearMipmapLinearBO       (const VirtualTextureKernelParams p) { VirtualTextureGradKernelTemplate<float, true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureGradKernelNearestHalf                (const VirtualTextureKernelParams p) { VirtualTextureGradKernelTemplate<half, false, TEX_MODE_NEAREST>(p); }
__global__ void VirtualTextureGradKernelLinearHalf                 (const VirtualTextureKernelParams p) { VirtualTextureGradKernelTemplate<half, false, TEX_MODE_LINEAR>(p); }
__global__ void VirtualTextureGradKernelLinearMipmapNearestHalf    (const VirtualTextureKernelParams p) { VirtualTextureGradKernelTemplate<half, false, TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureGradKernelLinearMipmapLinearHalf     (const VirtualTextureKernelParams p) { VirtualTextureGradKernelTemplate<half, false, TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }
__global__ void VirtualTextureGradKernelLinearMipmapNearestBOHalf  (const VirtualTextureKernelParams p) { VirtualTextureGradKernelTemplate<half, true,  TEX_MODE_LINEAR_MIPMAP_NEAREST>(p); }
__global__ void VirtualTextureGradKernelLinearMipmapLinearBOHalf   (const VirtualTextureKernelParams p) { VirtualTextureGradKernelTemplate<half, true,  TEX_MODE_LINEAR_MIPMAP_LINEAR>(p); }

static __forceinline__ __device__ int2 indexPage(int px, int py, int width, int height, int page_size_x, int page_size_y)
{
    // Because sometimes the width and height may be smaller than the page size,
    // so we need to decrease the page_size_x and page_size_y to width and height
    page_size_x = width >= page_size_x ? page_size_x : width;
    page_size_y = height >= page_size_y ? page_size_y : height;

    int iu = px % page_size_x;
    int iv = py % page_size_y;
    int tc = iv * page_size_x + iu;

    int page_x = px / page_size_x;
    int page_y = py / page_size_y;
    int page_num_x = width / page_size_x;
    int page_index = page_y * page_num_x + page_x;

    return make_int2(page_index, tc);
}

template <class T, int C>
static __forceinline__ __device__ void VirtualTextureMipmapTemplate(const VirtualTextureMipmapParams p)
{
    int out_width = p.texWidth / 2;     // Width of outgoing mipmap
    int out_height = p.texHeight / 2;   // Height of outgoing mipmap
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= out_width || py >= out_height)
        return;

    int2 pi_tc = indexPage(px, py, out_width, out_height, p.page_size_x, p.page_size_y);
    int page_index = pi_tc.x;
    int tc = pi_tc.y;

    tc *= p.channels;

    using value_type = typename value_type_traits<T>::type;
    value_type* pOut = (value_type*)p.out[page_index] + tc;

    const int offset_x[] = {0, 1, 
                            0, 1};
    const int offset_y[] = {1, 1, 
                            0, 0};

    for (int i=0; i < p.channels; i += C)
    {
        T texel = zero_value<T>();
        bool all_true = true;
        bool all_false = true;
        for (int j = 0; j < 4; j++)
        {
            int px_in = px*2 + offset_x[j];
            int py_in = py*2 + offset_y[j];
            if (px >= p.texWidth || py >= p.texHeight)
                continue;

            int2 pi_tc_in = indexPage(px_in, py_in, p.texWidth, p.texHeight, p.page_size_x, p.page_size_y);
            int page_index_in = pi_tc_in.x;
            int tc_in = pi_tc_in.y;
            tc_in *= p.channels;
            if (p.tex[page_index_in] != NULL) {
                all_false = false;
            } else {
                all_true = false;
            }
            if (p.tex[page_index_in] != NULL)
            {
                const value_type* pIn = (value_type*)p.tex[page_index_in] + tc_in;
                texel += *((const T*)&pIn[i]);
            }
        }
        *((T*)&pOut[i]) = texel * cast<T>(0.25f);
    }

}

__global__ void VirtualTextureMipmapKernel1                    (const VirtualTextureMipmapParams p) { VirtualTextureMipmapTemplate<float,  1>(p); }
__global__ void VirtualTextureMipmapKernel2                    (const VirtualTextureMipmapParams p) { VirtualTextureMipmapTemplate<float2,  2>(p); }
__global__ void VirtualTextureMipmapKernel4                    (const VirtualTextureMipmapParams p) { VirtualTextureMipmapTemplate<float4,  4>(p); }
__global__ void VirtualTextureMipmapKernelHalf1                (const VirtualTextureMipmapParams p) { VirtualTextureMipmapTemplate<half,  1>(p); }
__global__ void VirtualTextureMipmapKernelHalf2                (const VirtualTextureMipmapParams p) { VirtualTextureMipmapTemplate<half2,  2>(p); }
__global__ void VirtualTextureMipmapKernelHalf4                (const VirtualTextureMipmapParams p) { VirtualTextureMipmapTemplate<half4,  4>(p); }


//------------------------------------------------------------------------
// Gradient mip puller kernel.

template<class T, int C>
static __forceinline__ __device__ void VirtualTextureMipGradKernelTemplate(const VirtualTextureKernelParams p, int pixel_num_x, int pixel_num_y)
{
    using value_type = typename value_type_traits<T>::type;
    int base_px = (blockIdx.x * blockDim.x + threadIdx.x) * pixel_num_x;
    int base_py = (blockIdx.y * blockDim.y + threadIdx.y) * pixel_num_y;

    // Number of wide elements.
    int c = p.channels;
    if (C == 2) c >>= 1;
    if (C == 4) c >>= 2;

    // Dynamically allocated shared memory for holding a texel.
    extern __shared__ float s_texelAccum[];
    int sharedOfs = threadIdx.x + threadIdx.y * blockDim.x;
    int sharedStride = blockDim.x * blockDim.y;
#   define TEXEL_ACCUM(_i) (s_texelAccum + (sharedOfs + (_i) * sharedStride))

    for (int pixel_x=0; pixel_x<pixel_num_x; pixel_x++)
    {
        for (int pixel_y=0; pixel_y<pixel_num_y; pixel_y++)
        {
            // Clear the texel.
            for (int i=0; i < p.channels; i++)
                *TEXEL_ACCUM(i) = 0.f;

            // Calculate pixel position.
            int px = base_px + pixel_x;
            int py = base_py + pixel_y;
            int pz = blockIdx.z;
            if (px >= p.texWidth || py >= p.texHeight)
                continue;
            int2 sz = mipLevelSize(p, 0);
            int2 pi_tc = indexPage(px, py, sz.x, sz.y, p.page_size_x, p.page_size_y);
            int pi = pi_tc.x; int tc = pi_tc.y;
            if (p.gradTex[0][pi] == NULL)
                continue;
                
            // Track texel position and accumulation weight over the mip stack.
            int x = px;
            int y = py;
            float w = 1.f;

            // Pull gradients from all levels.
            for (int level=1; level <= p.mipLevelMax; level++)
            {
                // Weight decay depends on previous level size.
                if (sz.x > 1) w *= .5f;
                if (sz.y > 1) w *= .5f;

                // Current level size and coordinates.
                sz = mipLevelSize(p, level);
                x >>= 1;
                y >>= 1;

                int2 pi_tc = indexPage(x, y, sz.x, sz.y, p.page_size_x, p.page_size_y);
                int pi = pi_tc.x; int tc = pi_tc.y;

                if (p.gradTex[level][pi] == NULL)
                    continue;
                
                T* pIn = (T*)((value_type*)p.gradTex[level][pi] + tc * p.channels);
                for (int i=0; i < c; i++)
                {
                    if constexpr (C == 1)
                        accum_from_mem(TEXEL_ACCUM(i * C), sharedStride, cast<float>(pIn[i]), w);
                    else if constexpr (C == 2)
                        accum_from_mem(TEXEL_ACCUM(i * C), sharedStride, cast<float2>(pIn[i]), w);
                    else if constexpr (C == 4)
                        accum_from_mem(TEXEL_ACCUM(i * C), sharedStride, cast<float4>(pIn[i]), w);
                }
            }

            // Add to main texture gradients.
            T* pOut = (T*)((value_type*)p.gradTex[0][pi] + tc * p.channels);
            for (int i=0; i < c; i++)
                accum_to_mem(pOut[i], TEXEL_ACCUM(i * C), sharedStride);
        }
    }
}

// Template specializations.
__global__ void VirtualTextureMipGradKernel1(const VirtualTextureKernelParams p, int pixel_num_x, int pixel_num_y) { VirtualTextureMipGradKernelTemplate<float,  1>(p, pixel_num_x, pixel_num_y); }
__global__ void VirtualTextureMipGradKernel2(const VirtualTextureKernelParams p, int pixel_num_x, int pixel_num_y) { VirtualTextureMipGradKernelTemplate<float2, 2>(p, pixel_num_x, pixel_num_y); }
__global__ void VirtualTextureMipGradKernel4(const VirtualTextureKernelParams p, int pixel_num_x, int pixel_num_y) { VirtualTextureMipGradKernelTemplate<float4, 4>(p, pixel_num_x, pixel_num_y); }
__global__ void VirtualTextureMipGradKernelHalf1(const VirtualTextureKernelParams p, int pixel_num_x, int pixel_num_y) { VirtualTextureMipGradKernelTemplate<half,  1>(p, pixel_num_x, pixel_num_y); }
__global__ void VirtualTextureMipGradKernelHalf2(const VirtualTextureKernelParams p, int pixel_num_x, int pixel_num_y) { VirtualTextureMipGradKernelTemplate<half2, 2>(p, pixel_num_x, pixel_num_y); }
__global__ void VirtualTextureMipGradKernelHalf4(const VirtualTextureKernelParams p, int pixel_num_x, int pixel_num_y) { VirtualTextureMipGradKernelTemplate<half4, 4>(p, pixel_num_x, pixel_num_y); }
