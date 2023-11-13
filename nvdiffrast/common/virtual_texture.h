
#pragma once
#include "texture.h"

//------------------------------------------------------------------------
// CUDA kernel params.

struct VirtualTextureKernelParams
{
    const float**   tex[TEX_MAX_MIP_LEVEL];         // Incoming texture pages.
    const float*    uv;                             // Incoming texcoord buffer.
    const float*    uvDA;                           // Incoming uv pixel diffs or NULL.
    const float*    mipLevelBias;                   // Incoming mip level bias or NULL.
    const float*    dy;                             // Incoming output gradient.
    float*          out;                            // Outgoing texture data.
    float**         gradTex[TEX_MAX_MIP_LEVEL];     // Outgoing texture gradients pages.
    float*          gradUV;                         // Outgoing texcoord gradient.
    float*          gradUVDA;                       // Outgoing texcoord pixel differential gradient.
    float*          gradMipLevelBias;               // Outgoing mip level bias gradient.
    unsigned char*  feedback[TEX_MAX_MIP_LEVEL];    // Outgoing virtual texture pages feedback.
    int             enableMip;                      // If true, we have uv_da and/or mip_level_bias input(s), and a mip tensor.
    int             filterMode;                     // One of the TEX_MODE_ constants.
    int             boundaryMode;                   // One of the TEX_BOUNDARY_MODE_ contants.
    int             texConst;                       // If true, texture is known to be constant.
    // int             mipLevelLimit;                  // Mip level limit coming from the op.
    int             channels;                       // Number of texture channels.
    int             imgWidth;                       // Image width.
    int             imgHeight;                      // Image height.
    int             texWidth;                       // Texture width.
    int             texHeight;                      // Texture height.
    int             texDepth;                       // Texture depth.
    int             n;                              // Minibatch size.
    int             mipLevelMax;                    // Maximum mip level index. Zero if mips disabled.
    int             mipLevelOut;                    // Mip level being calculated in builder kernel.
    int             page_size_x;                    // Number of pixels of a virtual texture page in x axis
    int             page_size_y;                    // Number of pixels of a virtual texture page in y axis
};

// struct VirtualTextureFeedbackParams
// {
//     const float*    uv;                             // Incoming texcoord buffer.
//     const float*    uvDA;                           // Incoming uv pixel diffs or NULL.
//     const float*    mipLevelBias;                   // Incoming mip level bias or NULL.
//     unsigned char*  out[TEX_MAX_MIP_LEVEL];         // Outgoing virtual texture pages feedback.
//     int             enableMip;                      // If true, we have uv_da and/or mip_level_bias input(s), and a mip tensor.
//     int             filterMode;                     // One of the TEX_MODE_ constants.
//     int             boundaryMode;                   // One of the TEX_BOUNDARY_MODE_ contants.
//     int             channels;                       // Number of incomming texture channels.
//     int             imgWidth;                       // Image width.
//     int             imgHeight;                      // Image height.
//     int             texWidth;                       // Incomming texture width.
//     int             texHeight;                      // Incomming texture height.
//     int             texDepth;                       // Incomming Texture depth.
//     int             n;                              // Minibatch size.
//     int             mipLevelMax;                    // Maximum mip level index. Zero if mips disabled.
//     int             page_size_x;                    // Number of pixels of a virtual texture page in x axis
//     int             page_size_y;                    // Number of pixels of a virtual texture page in y axis
// };

struct VirtualTextureMipmapParams
{
    float const**   tex;                            // Incoming texture mipmap pages.
    float**         out;                            // Outgoing texture mipmap pages.
    int             channels;                       // Number of incomming texture channels.
    int             texWidth;                       // Incomming texture width.
    int             texHeight;                      // Incomming texture height.
    int             texDepth;                       // Incomming Texture depth.
    int             page_size_x;                    // Number of pixels of a virtual texture page in x axis
    int             page_size_y;                    // Number of pixels of a virtual texture page in y axis
};

#define calcPageNum(wh, page_size) (((wh) < (page_size)) ? 1 : ((wh) / (page_size)))