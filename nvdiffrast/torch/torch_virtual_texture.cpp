
#include "torch_common.inl"
#include "torch_types.h"
#include "../common/common.h"
#include "../common/virtual_texture.h"
#include "../common/parallel.hpp"
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <type_traits>

//------------------------------------------------------------------------
// Kernel prototypes.

void VirtualTextureFeedbackKernelNearest1                   (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelNearest2                   (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelNearest4                   (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelLinear1                    (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelLinear2                    (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelLinear4                    (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelLinearMipmapNearest1       (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelLinearMipmapNearest2       (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelLinearMipmapNearest4       (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelLinearMipmapLinear1        (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelLinearMipmapLinear2        (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelLinearMipmapLinear4        (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelLinearMipmapNearestBO1     (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelLinearMipmapNearestBO2     (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelLinearMipmapNearestBO4     (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelLinearMipmapLinearBO1      (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelLinearMipmapLinearBO2      (const VirtualTextureKernelParams p);
void VirtualTextureFeedbackKernelLinearMipmapLinearBO4      (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelNearest1                        (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelNearest2                        (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelNearest4                        (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinear1                         (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinear2                         (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinear4                         (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapNearest1            (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapNearest2            (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapNearest4            (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapLinear1             (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapLinear2             (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapLinear4             (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapNearestBO1          (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapNearestBO2          (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapNearestBO4          (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapLinearBO1           (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapLinearBO2           (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapLinearBO4           (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelNearestHalf1                        (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelNearestHalf2                        (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelNearestHalf4                        (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearHalf1                         (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearHalf2                         (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearHalf4                         (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapNearestHalf1            (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapNearestHalf2            (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapNearestHalf4            (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapLinearHalf1             (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapLinearHalf2             (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapLinearHalf4             (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapNearestBOHalf1          (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapNearestBOHalf2          (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapNearestBOHalf4          (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapLinearBOHalf1           (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapLinearBOHalf2           (const VirtualTextureKernelParams p);
void VirtualTextureFwdKernelLinearMipmapLinearBOHalf4           (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelNearest                        (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelLinear                         (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelLinearMipmapNearest            (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelLinearMipmapLinear             (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelLinearMipmapNearestBO          (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelLinearMipmapLinearBO           (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelNearestHalf                        (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelLinearHalf                         (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelLinearMipmapNearestHalf            (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelLinearMipmapLinearHalf             (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelLinearMipmapNearestBOHalf          (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelLinearMipmapLinearBOHalf           (const VirtualTextureKernelParams p);
void VirtualTextureMipmapKernel1                            (const VirtualTextureMipmapParams p);
void VirtualTextureMipmapKernel2                            (const VirtualTextureMipmapParams p);
void VirtualTextureMipmapKernel4                            (const VirtualTextureMipmapParams p);
void VirtualTextureMipmapKernelHalf1                            (const VirtualTextureMipmapParams p);
void VirtualTextureMipmapKernelHalf2                            (const VirtualTextureMipmapParams p);
void VirtualTextureMipmapKernelHalf4                            (const VirtualTextureMipmapParams p);
void VirtualTextureMipGradKernel1                           (const VirtualTextureKernelParams p, int, int);
void VirtualTextureMipGradKernel2                           (const VirtualTextureKernelParams p, int, int);
void VirtualTextureMipGradKernel4                           (const VirtualTextureKernelParams p, int, int);
void VirtualTextureMipGradKernelHalf1                           (const VirtualTextureKernelParams p, int, int);
void VirtualTextureMipGradKernelHalf2                           (const VirtualTextureKernelParams p, int, int);
void VirtualTextureMipGradKernelHalf4                           (const VirtualTextureKernelParams p, int, int);

int calculateMaxMipLevel(int width, int height, int mipLevelLimit)
{

    if (mipLevelLimit == 0)
        return 0;

    int w = width;
    int h = height;

    int level = 0;
    while ((w|h) > 1)
    {
        // Current level.
        level += 1;

        // Downsample.
        if (w > 1) w >>= 1;
        if (h > 1) h >>= 1;

        if (mipLevelLimit >= 0 && level == mipLevelLimit)
            break;
    }

    return level;
}

#define calcMipLevelSize(w, h, i) make_int2((w >> (i)) > 1 ? (w >> (i)) : 1, (h >> (i)) > 1 ? (h >> (i)) : 1)

//------------------------------------------------------------------------
// Modeselektor.

static void set_modes(VirtualTextureKernelParams& p, int filter_mode, int boundary_mode, int max_mip_level)
{
    // Mip and filter modes.
    p.filterMode = filter_mode;
    NVDR_CHECK(p.filterMode >= 0 && p.filterMode < TEX_MODE_COUNT, "filter_mode unsupported");
    p.enableMip = (p.filterMode == TEX_MODE_LINEAR_MIPMAP_NEAREST || p.filterMode == TEX_MODE_LINEAR_MIPMAP_LINEAR);

    // Mip level clamp.
    // if (p.enableMip)
    // {
    //     p.mipLevelLimit = max_mip_level;
    //     NVDR_CHECK(p.mipLevelLimit >= -1, "invalid max_mip_level");
    // }

    // Boundary mode.
    p.boundaryMode = boundary_mode;
    NVDR_CHECK(p.boundaryMode >= 0 && p.boundaryMode < TEX_BOUNDARY_MODE_COUNT, "boundary_mode unsupported");
}

//------------------------------------------------------------------------
// Virtual texture feedback op.

std::vector<torch::Tensor> virtual_texture_feedback_mip(torch::Tensor uv, torch::Tensor uv_da, torch::Tensor mip_level_bias, torch::Tensor mask, int filter_mode, int boundary_mode, int texture_depth, int texture_height, int texture_width, int texture_channels, int page_size_x, int page_size_y, int max_mip_level)
{
    NVDR_CHECK(texture_height>0 && (texture_height & (texture_height-1))==0, "virtual_texture_feedback_mip: Texture height must be power of two.");
    NVDR_CHECK(texture_width>0 && (texture_width & (texture_width-1))==0, "virtual_texture_feedback_mip: Texture width must be power of two.");
    NVDR_CHECK(page_size_y>0 && (page_size_y & (page_size_y-1))==0, "virtual_texture_feedback_mip: Page Y must be power of two.");
    NVDR_CHECK(page_size_x>0 && (page_size_x & (page_size_x-1))==0, "virtual_texture_feedback_mip: Page X must be power of two.");
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(uv));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    VirtualTextureKernelParams p = {}; // Initialize all fields to zero.
    max_mip_level = calculateMaxMipLevel(texture_width, texture_height, max_mip_level);
    set_modes(p, filter_mode, boundary_mode, max_mip_level);
    p.mipLevelMax = max_mip_level;

    bool has_mask = mask.defined() && mask.nbytes();
    p.mask = has_mask ? mask.data_ptr<bool>() : NULL;

    // See if we have these tensors or not.
    bool has_uv_da = uv_da.defined() && uv_da.nbytes();
    bool has_mip_level_bias = mip_level_bias.defined() && mip_level_bias.nbytes();

    if (p.enableMip)
    {
        NVDR_CHECK(has_uv_da || has_mip_level_bias, "mipmapping filter mode requires uv_da and/or mip_level_bias input");
    }

    // Check inputs.
    NVDR_CHECK_DEVICE(uv);
    NVDR_CHECK_CONTIGUOUS(uv);
    NVDR_CHECK_F32(uv);
    if (p.enableMip)
    {
        if (has_uv_da)
        {
            NVDR_CHECK_DEVICE(uv_da);
            NVDR_CHECK_CONTIGUOUS(uv_da);
            NVDR_CHECK_F32(uv_da);
        }
        if (has_mip_level_bias)
        {
            NVDR_CHECK_DEVICE(mip_level_bias);
            NVDR_CHECK_CONTIGUOUS(mip_level_bias);
            NVDR_CHECK_F32(mip_level_bias);
        }
    }

    // Sanity checks and state setters.
    p.texHeight = texture_height;
    p.texWidth  = texture_width;
    p.channels  = texture_channels;
    NVDR_CHECK(texture_depth == 1 || texture_depth == uv.size(0), "minibatch size mismatch between inputs tex, uv");
    NVDR_CHECK(p.texWidth <= (1 << TEX_MAX_MIP_LEVEL) && p.texHeight <= (1 << TEX_MAX_MIP_LEVEL), "texture size too large");
    p.n         = uv.size(0);
    p.imgHeight = uv.size(1);
    p.imgWidth  = uv.size(2);
    p.texDepth  = texture_depth;
    if (p.enableMip)
    {
        if (has_uv_da)
            NVDR_CHECK(uv_da.sizes().size() == 4 && uv_da.size(0) == p.n && uv_da.size(1) == p.imgHeight && uv_da.size(2) == p.imgWidth && uv_da.size(3) == 4, "uv_da must have shape [minibatch_size, height, width, 4]");
        if (has_mip_level_bias)
            NVDR_CHECK(mip_level_bias.sizes().size() == 3 && mip_level_bias.size(0) == p.n && mip_level_bias.size(1) == p.imgHeight && mip_level_bias.size(2) == p.imgWidth, "mip_level_bias must have shape [minibatch_size, height, width]");
    }

    // Get input pointers.
    p.uv = uv.data_ptr<float>();
    p.uvDA = (p.enableMip && has_uv_da) ? uv_da.data_ptr<float>() : NULL;
    p.mipLevelBias = (p.enableMip && has_mip_level_bias) ? mip_level_bias.data_ptr<float>() : NULL;
    p.page_size_x = page_size_x;
    p.page_size_y = page_size_y;


    std::vector<torch::Tensor> Out(max_mip_level+1);
    for (int mip = 0; mip <= max_mip_level; mip++)
    {
        int2 sz_mip = calcMipLevelSize(texture_width, texture_height, mip);
        int width_mip = sz_mip.x;
        int height_mip = sz_mip.y;
        int page_num_y_mip = calcPageNum(height_mip, page_size_y);
        int page_num_x_mip = calcPageNum(width_mip, page_size_x);
        int page_num_mip = page_num_y_mip * page_num_x_mip;
        int numPages = page_num_mip;
        // Allocate output tensor.
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
        torch::Tensor feedback = torch::zeros({numPages}, opts);
        Out[mip] = feedback;
        p.feedback[mip] = feedback.data_ptr<bool>();
    }

    // Choose kernel variants based on channel count.
    void* args[] = {&p};
    int channel_div_idx = 0;
    if (!(p.channels & 3))
        channel_div_idx = 2;  // Channel count divisible by 4.
    else if (!(p.channels & 1))
        channel_div_idx = 1;  // Channel count divisible by 2.

    // Choose launch parameters for texture lookup kernel.
    dim3 blockSize = getLaunchBlockSize(TEX_FWD_MAX_KERNEL_BLOCK_WIDTH, TEX_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.imgWidth, p.imgHeight);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.imgWidth, p.imgHeight, p.n);

    // Choose kernel based on filter mode, cube mode, bias-only mode, and datatype.
    void* func_tbl[TEX_MODE_COUNT * 2 * 3] = {
        (void*)VirtualTextureFeedbackKernelNearest1,
        (void*)VirtualTextureFeedbackKernelNearest2,
        (void*)VirtualTextureFeedbackKernelNearest4,
        (void*)VirtualTextureFeedbackKernelLinear1,
        (void*)VirtualTextureFeedbackKernelLinear2,
        (void*)VirtualTextureFeedbackKernelLinear4,
        (void*)VirtualTextureFeedbackKernelLinearMipmapNearest1,
        (void*)VirtualTextureFeedbackKernelLinearMipmapNearest2,
        (void*)VirtualTextureFeedbackKernelLinearMipmapNearest4,
        (void*)VirtualTextureFeedbackKernelLinearMipmapLinear1,
        (void*)VirtualTextureFeedbackKernelLinearMipmapLinear2,
        (void*)VirtualTextureFeedbackKernelLinearMipmapLinear4,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        (void*)VirtualTextureFeedbackKernelLinearMipmapNearestBO1,
        (void*)VirtualTextureFeedbackKernelLinearMipmapNearestBO2,
        (void*)VirtualTextureFeedbackKernelLinearMipmapNearestBO4,
        (void*)VirtualTextureFeedbackKernelLinearMipmapLinearBO1,
        (void*)VirtualTextureFeedbackKernelLinearMipmapLinearBO2,
        (void*)VirtualTextureFeedbackKernelLinearMipmapLinearBO4,
    };

    // Function index.
    int func_idx = p.filterMode;
    if (p.enableMip && !has_uv_da)
        func_idx += TEX_MODE_COUNT; // Bias-only variant.
    func_idx = func_idx * 3 + channel_div_idx; // Choose vector size.

    // Launch kernel.
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func_tbl[func_idx], gridSize, blockSize, args, 0, stream));

    // Return output tensor.
    return Out;

}

// Version without mipmaps.
std::vector<torch::Tensor> virtual_texture_feedback(torch::Tensor uv, torch::Tensor mask, int filter_mode, int boundary_mode, int texture_depth, int texture_height, int texture_width, int texture_channels, int page_size_x, int page_size_y)
{
    torch::Tensor empty_tensor;
    return virtual_texture_feedback_mip(uv, empty_tensor, empty_tensor, mask, filter_mode, boundary_mode, texture_depth, texture_height, texture_width, texture_channels, page_size_x, page_size_y, 0);
}

//------------------------------------------------------------------------
// Forward op.

// Check and get dtype of cuda pages
std::vector<torch::Tensor> check_and_get_type(std::vector<std::vector<torch::Tensor>> pages, torch::ScalarType& dtype)
{
    std::vector<torch::Tensor> cuda_pages;
    for (auto& mip_pages : pages)
    {
        for (auto& page : mip_pages)
        {
            if (at::cuda::check_device({page}))
            {
                cuda_pages.push_back(page);
            }
        }
    }
    if (!cuda_pages.empty())
    {
#ifdef ENABLE_HALF_TEXTURE
        NVDR_CHECK_F16F32(cuda_pages);
#else
        NVDR_CHECK_F32(cuda_pages);
#endif // ENABLE_HALF_TEXTURE
        NVDR_CHECK_CONTIGUOUS(cuda_pages);
        dtype = cuda_pages[0].dtype().toScalarType();
    }

    return cuda_pages;
}

torch::Tensor virtual_texture_fwd_mip(
    torch::Tensor uv, torch::Tensor uv_da, torch::Tensor mip_level_bias, torch::Tensor mask, 
    int filter_mode, int boundary_mode, 
    int texture_depth, int texture_height, int texture_width, int texture_channels,
    int page_size_x, int page_size_y, 
    std::vector<std::vector<torch::Tensor>> pages)
{
    NVDR_CHECK(texture_height>0 && (texture_height & (texture_height-1))==0, "virtual_texture_fwd_mip: Texture height must be power of two.");
    NVDR_CHECK(texture_width>0 && (texture_width & (texture_width-1))==0, "virtual_texture_fwd_mip: Texture width must be power of two.");
    NVDR_CHECK(page_size_y>0 && (page_size_y & (page_size_y-1))==0, "virtual_texture_fwd_mip: Page Y must be power of two.");
    NVDR_CHECK(page_size_x>0 && (page_size_x & (page_size_x-1))==0, "virtual_texture_fwd_mip: Page X must be power of two.");


    const at::cuda::OptionalCUDAGuard device_guard(device_of(uv));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    VirtualTextureKernelParams p = {}; // Initialize all fields to zero.
    int max_mip_level = pages.size()-1;
    set_modes(p, filter_mode, boundary_mode, max_mip_level);
    p.n         = uv.size(0);
    p.imgHeight = uv.size(1);
    p.imgWidth  = uv.size(2);
    p.texDepth  = texture_depth;
    p.texHeight = texture_height;
    p.texWidth  = texture_width;
    p.channels  = texture_channels;
    p.page_size_x = page_size_x;
    p.page_size_y = page_size_y;
    p.mipLevelMax = max_mip_level;

    bool has_mask = mask.defined() && mask.nbytes();
    p.mask = has_mask ? mask.data_ptr<bool>() : NULL;

    // See if we have these tensors or not.
    bool has_uv_da = uv_da.defined() && uv_da.nbytes();
    bool has_mip_level_bias = mip_level_bias.defined() && mip_level_bias.nbytes();

    if (p.enableMip)
    {
        NVDR_CHECK(has_uv_da || has_mip_level_bias, "mipmapping filter mode requires uv_da and/or mip_level_bias input");
    }

    // Check inputs.
    NVDR_CHECK_DEVICE(uv);
    NVDR_CHECK_CONTIGUOUS(uv);
    NVDR_CHECK_F32(uv);
    if (p.enableMip)
    {
        if (has_uv_da)
        {
            NVDR_CHECK_DEVICE(uv_da);
            NVDR_CHECK_CONTIGUOUS(uv_da);
            NVDR_CHECK_F32(uv_da);
            NVDR_CHECK(uv_da.sizes().size() == 4 && uv_da.size(0) == p.n && uv_da.size(1) == p.imgHeight && uv_da.size(2) == p.imgWidth && uv_da.size(3) == 4, "uv_da must have shape [minibatch_size, height, width, 4]");
        }
        if (has_mip_level_bias)
        {
            NVDR_CHECK_DEVICE(mip_level_bias);
            NVDR_CHECK_CONTIGUOUS(mip_level_bias);
            NVDR_CHECK_F32(mip_level_bias);
            NVDR_CHECK(mip_level_bias.sizes().size() == 3 && mip_level_bias.size(0) == p.n && mip_level_bias.size(1) == p.imgHeight && mip_level_bias.size(2) == p.imgWidth, "mip_level_bias must have shape [minibatch_size, height, width]");
        }
    }
    NVDR_CHECK(uv.sizes().size() == 4 && uv.size(0) > 0 && uv.size(1) > 0 && uv.size(2) > 0 && uv.size(3) == 2, "uv must have shape [>0, >0, >0, 2]");
    NVDR_CHECK(p.texWidth <= (1 << TEX_MAX_MIP_LEVEL) && p.texHeight <= (1 << TEX_MAX_MIP_LEVEL), "texture size too large");


    auto mip_ptr = prepareCudaTensorArray(pages);
    for (int mip = 0; mip <= max_mip_level; mip++)
    {
        int2 sz = calcMipLevelSize(texture_width, texture_height, mip);
        int page_num_y = calcPageNum(sz.y, page_size_y);
        int page_num_x = calcPageNum(sz.x, page_size_x);
        int numPages = page_num_x*page_num_y;
        NVDR_CHECK(numPages == pages[mip].size(), "virtual_texture_fwd_mip: The number of pages mismatches in mipmap level " + std::to_string(mip));
        p.tex[mip] = (const float**)mip_ptr[mip].data_ptr();
    }
    p.uv = uv.data_ptr<float>();
    p.uvDA = (p.enableMip && has_uv_da) ? uv_da.data_ptr<float>() : NULL;
    p.mipLevelBias = (p.enableMip && has_mip_level_bias) ? mip_level_bias.data_ptr<float>() : NULL;

    // Allocate output tensor.
    torch::ScalarType dtype;
    bool has_cuda_pages = !check_and_get_type(pages, dtype).empty();
    torch::TensorOptions opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
    torch::Tensor out = torch::zeros({p.n, p.imgHeight, p.imgWidth, p.channels}, opts);
    if (!has_cuda_pages)
        return out;
    p.out = (float*)out.data_ptr();

    // Choose kernel variants based on channel count.
    void* args[] = {&p};
    int channel_div_idx = 0;
    if (!(p.channels & 3))
        channel_div_idx = 2;  // Channel count divisible by 4.
    else if (!(p.channels & 1))
        channel_div_idx = 1;  // Channel count divisible by 2.

    // Verify that buffers are aligned to allow float2/float4 operations. Unused pointers are zero so always aligned.
    NVDR_CHECK(!((uintptr_t)p.uv & 7), "uv input tensor not aligned to float2");
    if ((p.channels & 3) == 0)
    {
        for (int i=0; i <= p.mipLevelMax; i++)
        {
            for (int j=0; j < pages[i].size(); j++)
                NVDR_CHECK(!((uintptr_t)p.tex[i][j] & 15), "tex or mip input tensor not aligned to float4");
        }
        NVDR_CHECK(!((uintptr_t)p.out    & 15), "out output tensor not aligned to float4");
    }
    if ((p.channels & 1) == 0)
    {
        for (int i=0; i <= p.mipLevelMax; i++)
        {
            for (int j=0; j < pages[i].size(); j++)
                NVDR_CHECK(!((uintptr_t)p.tex[i][j] & 7), "tex or mip input tensor not aligned to float2");
        }
        NVDR_CHECK(!((uintptr_t)p.out    & 7), "out output tensor not aligned to float2");
    }
    NVDR_CHECK(!((uintptr_t)p.uvDA & 15), "uv_da input tensor not aligned to float4");

    // Choose launch parameters for texture lookup kernel.
    dim3 blockSize = getLaunchBlockSize(TEX_FWD_MAX_KERNEL_BLOCK_WIDTH, TEX_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.imgWidth, p.imgHeight);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.imgWidth, p.imgHeight, p.n);

    // Choose kernel based on filter mode, cube mode, bias-only mode, and datatype.
    void* func_tbl[TEX_MODE_COUNT * 2 * 3 * 2] = {
        (void*)VirtualTextureFwdKernelNearest1,
        (void*)VirtualTextureFwdKernelNearest2,
        (void*)VirtualTextureFwdKernelNearest4,
        (void*)VirtualTextureFwdKernelLinear1,
        (void*)VirtualTextureFwdKernelLinear2,
        (void*)VirtualTextureFwdKernelLinear4,
        (void*)VirtualTextureFwdKernelLinearMipmapNearest1,
        (void*)VirtualTextureFwdKernelLinearMipmapNearest2,
        (void*)VirtualTextureFwdKernelLinearMipmapNearest4,
        (void*)VirtualTextureFwdKernelLinearMipmapLinear1,
        (void*)VirtualTextureFwdKernelLinearMipmapLinear2,
        (void*)VirtualTextureFwdKernelLinearMipmapLinear4,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        (void*)VirtualTextureFwdKernelLinearMipmapNearestBO1,
        (void*)VirtualTextureFwdKernelLinearMipmapNearestBO2,
        (void*)VirtualTextureFwdKernelLinearMipmapNearestBO4,
        (void*)VirtualTextureFwdKernelLinearMipmapLinearBO1,
        (void*)VirtualTextureFwdKernelLinearMipmapLinearBO2,
        (void*)VirtualTextureFwdKernelLinearMipmapLinearBO4,
        (void*)VirtualTextureFwdKernelNearestHalf1,
        (void*)VirtualTextureFwdKernelNearestHalf2,
        (void*)VirtualTextureFwdKernelNearestHalf4,
        (void*)VirtualTextureFwdKernelLinearHalf1,
        (void*)VirtualTextureFwdKernelLinearHalf2,
        (void*)VirtualTextureFwdKernelLinearHalf4,
        (void*)VirtualTextureFwdKernelLinearMipmapNearestHalf1,
        (void*)VirtualTextureFwdKernelLinearMipmapNearestHalf2,
        (void*)VirtualTextureFwdKernelLinearMipmapNearestHalf4,
        (void*)VirtualTextureFwdKernelLinearMipmapLinearHalf1,
        (void*)VirtualTextureFwdKernelLinearMipmapLinearHalf2,
        (void*)VirtualTextureFwdKernelLinearMipmapLinearHalf4,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        (void*)VirtualTextureFwdKernelLinearMipmapNearestBOHalf1,
        (void*)VirtualTextureFwdKernelLinearMipmapNearestBOHalf2,
        (void*)VirtualTextureFwdKernelLinearMipmapNearestBOHalf4,
        (void*)VirtualTextureFwdKernelLinearMipmapLinearBOHalf1,
        (void*)VirtualTextureFwdKernelLinearMipmapLinearBOHalf2,
        (void*)VirtualTextureFwdKernelLinearMipmapLinearBOHalf4,
    };

    // Function index.
    int func_idx = p.filterMode;
    if (p.enableMip && !has_uv_da)
        func_idx += TEX_MODE_COUNT; // Bias-only variant.
    func_idx = func_idx * 3 + channel_div_idx; // Choose vector size.
    if (dtype == torch::kHalf)
        func_idx += TEX_MODE_COUNT * 2 * 3; // Choose half variant.

    // Launch kernel.
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func_tbl[func_idx], gridSize, blockSize, args, 0, stream));

    // Return output tensor.
    return out;
}

// Version without mipmaps.
torch::Tensor virtual_texture_fwd(
    torch::Tensor uv, torch::Tensor mask, 
    int filter_mode, int boundary_mode,
    int texture_depth, int texture_height, int texture_width, int texture_channels,
    int page_size_x, int page_size_y, 
    std::vector<torch::Tensor> pages)
{
    torch::Tensor empty_tensor;
    std::vector<torch::Tensor> empty_vector;
    return virtual_texture_fwd_mip(
        uv, empty_tensor, empty_tensor, mask,
        filter_mode, boundary_mode,
        texture_depth, texture_height, texture_width, texture_channels,
        page_size_x, page_size_y, 
        {pages});
}

//------------------------------------------------------------------------
// Gradient op.

std::tuple<std::vector<std::vector<torch::Tensor>>, torch::Tensor, torch::Tensor, torch::Tensor > 
virtual_texture_grad_linear_mipmap_linear(torch::Tensor uv, torch::Tensor dy, torch::Tensor uv_da, torch::Tensor mip_level_bias, torch::Tensor mask, int filter_mode, int boundary_mode, int texture_depth, int texture_height, int texture_width, int texture_channels, int page_size_x, int page_size_y, std::vector<std::vector<torch::Tensor>> pages)
{
    NVDR_CHECK(texture_height>0 && (texture_height & (texture_height-1))==0, "virtual_texture_fwd_mip: Texture height must be power of two.");
    NVDR_CHECK(texture_width>0 && (texture_width & (texture_width-1))==0, "virtual_texture_fwd_mip: Texture width must be power of two.");
    NVDR_CHECK(page_size_y>0 && (page_size_y & (page_size_y-1))==0, "virtual_texture_fwd_mip: Page Y must be power of two.");
    NVDR_CHECK(page_size_x>0 && (page_size_x & (page_size_x-1))==0, "virtual_texture_fwd_mip: Page X must be power of two.");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(uv));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    VirtualTextureKernelParams p = {}; // Initialize all fields to zero.
    int max_mip_level = pages.size()-1;
    set_modes(p, filter_mode, boundary_mode, max_mip_level);
    p.n         = uv.size(0);
    p.imgHeight = uv.size(1);
    p.imgWidth  = uv.size(2);
    p.texDepth  = texture_depth;
    p.texHeight = texture_height;
    p.texWidth  = texture_width;
    p.channels  = texture_channels;
    p.page_size_x = page_size_x;
    p.page_size_y = page_size_y;
    p.mipLevelMax = max_mip_level;

    bool has_mask = mask.defined() && mask.nbytes();
    p.mask = has_mask ? mask.data_ptr<bool>() : NULL;

    // See if we have these tensors or not.
    bool has_uv_da = uv_da.defined() && uv_da.nbytes();
    bool has_mip_level_bias = mip_level_bias.defined() && mip_level_bias.nbytes();

    if (p.enableMip)
    {
        NVDR_CHECK(has_uv_da || has_mip_level_bias, "mipmapping filter mode requires uv_da and/or mip_level_bias input");
    }

    // Check inputs.
    NVDR_CHECK_DEVICE(uv);
    NVDR_CHECK_CONTIGUOUS(uv);
    NVDR_CHECK_F32(uv);
    torch::ScalarType dtype;
    bool has_cuda_pages = !check_and_get_type(pages, dtype).empty();
    if (p.enableMip)
    {
        if (has_uv_da)
        {
            NVDR_CHECK_DEVICE(uv_da);
            NVDR_CHECK_CONTIGUOUS(uv_da);
            NVDR_CHECK_F32(uv_da);
            NVDR_CHECK(uv_da.sizes().size() == 4 && uv_da.size(0) == p.n && uv_da.size(1) == p.imgHeight && uv_da.size(2) == p.imgWidth && uv_da.size(3) == 4, "uv_da must have shape [minibatch_size, height, width, 4]");
        }
        if (has_mip_level_bias)
        {
            NVDR_CHECK_DEVICE(mip_level_bias);
            NVDR_CHECK_CONTIGUOUS(mip_level_bias);
            NVDR_CHECK_F32(mip_level_bias);
            NVDR_CHECK(mip_level_bias.sizes().size() == 3 && mip_level_bias.size(0) == p.n && mip_level_bias.size(1) == p.imgHeight && mip_level_bias.size(2) == p.imgWidth, "mip_level_bias must have shape [minibatch_size, height, width]");
        }
    }
    NVDR_CHECK(uv.sizes().size() == 4 && uv.size(0) > 0 && uv.size(1) > 0 && uv.size(2) > 0 && uv.size(3) == 2, "uv must have shape [>0, >0, >0, 2]");
    NVDR_CHECK(p.texWidth <= (1 << TEX_MAX_MIP_LEVEL) && p.texHeight <= (1 << TEX_MAX_MIP_LEVEL), "texture size too large");

    int page_num_y = calcPageNum(texture_height, page_size_y);
    int page_num_x = calcPageNum(texture_width, page_size_x);
    int numPages = page_num_x*page_num_y;

    // Get contiguous version of dy.
    torch::Tensor dy_ = dy.contiguous();

    auto mip_ptr = prepareCudaTensorArray(pages);
    for (int mip=0; mip <= max_mip_level; mip++)
    {
        p.tex[mip] = (const float**)mip_ptr[mip].data_ptr();
    }
    p.uv = uv.data_ptr<float>();
    p.dy = (float*)dy_.data_ptr();
    p.uvDA = (p.enableMip && has_uv_da) ? uv_da.data_ptr<float>() : NULL;
    p.mipLevelBias = (p.enableMip && has_mip_level_bias) ? mip_level_bias.data_ptr<float>() : NULL;

    // Allocate output tensor for tex gradient.
    std::vector<std::vector<torch::Tensor>> grad_tex;
    for (int mip = 0; mip <= max_mip_level; mip++)
    {
        int num_pages = pages[mip].size();
        std::vector<torch::Tensor> mip_grad_tex(num_pages);
        for (int page_index = 0; page_index < num_pages; page_index++)
        {
            if (pages[mip][page_index].defined() && pages[mip][page_index].nbytes() && pages[mip][page_index].is_cuda())
            {
                torch::Tensor grad_page = torch::zeros_like(pages[mip][page_index]);
                mip_grad_tex[page_index] = grad_page;
            }
        }
        grad_tex.push_back(mip_grad_tex);
    }
    auto grad_mip_ptr = prepareCudaTensorArray<false>(grad_tex);
    for (int mip = 0; mip <= max_mip_level; mip++)
    {
        p.gradTex[mip] = (float**)grad_mip_ptr[mip].data_ptr();
    }

    // Allocate output tensor for uv gradient.
    torch::Tensor grad_uv;
    torch::Tensor grad_uv_da;
    torch::Tensor grad_mip_level_bias;
    if (p.filterMode != TEX_MODE_NEAREST)
    {
        grad_uv = torch::zeros_like(uv);
        p.gradUV = grad_uv.data_ptr<float>();

        // Gradients for things affecting mip level.
        if (p.filterMode == TEX_MODE_LINEAR_MIPMAP_LINEAR)
        {
            // Allocate output tensor for uv_da gradient.
            if (has_uv_da)
            {
                grad_uv_da = torch::zeros_like(uv_da);
                p.gradUVDA = grad_uv_da.data_ptr<float>();
            }

            // Allocate output tensor for mip_level_bias gradient.
            if (has_mip_level_bias)
            {
                grad_mip_level_bias = torch::zeros_like(mip_level_bias);
                p.gradMipLevelBias = grad_mip_level_bias.data_ptr<float>();
            }
        }
    }

    // Choose kernel variants based on channel count.
    int channel_div_idx = 0;
    if (!(p.channels & 3))
        channel_div_idx = 2;  // Channel count divisible by 4.
    else if (!(p.channels & 1))
        channel_div_idx = 1;  // Channel count divisible by 2.

    // Verify that buffers are aligned to allow float2/float4 operations. Unused pointers are zero so always aligned.
    NVDR_CHECK(!((uintptr_t)p.uv & 7), "uv input tensor not aligned to float2");
    if ((p.channels & 3) == 0)
    {
        for (int i=0; i <= p.mipLevelMax; i++)
        {
            for (int j=0; j < pages[i].size(); j++)
                NVDR_CHECK(!((uintptr_t)p.tex[i][j] & 15), "tex or mip input tensor not aligned to float4");
        }
        NVDR_CHECK(!((uintptr_t)p.out    & 15), "out output tensor not aligned to float4");
    }
    if ((p.channels & 1) == 0)
    {
        for (int i=0; i <= p.mipLevelMax; i++)
        {
            for (int j=0; j < pages[i].size(); j++)
                NVDR_CHECK(!((uintptr_t)p.tex[i][j] & 7), "tex or mip input tensor not aligned to float2");
        }
        NVDR_CHECK(!((uintptr_t)p.out    & 7), "out output tensor not aligned to float2");
    }
    NVDR_CHECK(!((uintptr_t)p.uvDA & 15), "uv_da input tensor not aligned to float4");
    
    // Choose launch parameters for main gradient kernel.
    void* args[] = {&p};
    dim3 blockSize = getLaunchBlockSize(TEX_GRAD_MAX_KERNEL_BLOCK_WIDTH, TEX_GRAD_MAX_KERNEL_BLOCK_HEIGHT, p.imgWidth, p.imgHeight);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.imgWidth, p.imgHeight, p.n);

    void* func_tbl[TEX_MODE_COUNT * 2 * 2] = {
        (void*)VirtualTextureGradKernelNearest,
        (void*)VirtualTextureGradKernelLinear,
        (void*)VirtualTextureGradKernelLinearMipmapNearest,
        (void*)VirtualTextureGradKernelLinearMipmapLinear,
        NULL,
        NULL,
        (void*)VirtualTextureGradKernelLinearMipmapNearestBO,
        (void*)VirtualTextureGradKernelLinearMipmapLinearBO,
        (void*)VirtualTextureGradKernelNearestHalf,
        (void*)VirtualTextureGradKernelLinearHalf,
        (void*)VirtualTextureGradKernelLinearMipmapNearestHalf,
        (void*)VirtualTextureGradKernelLinearMipmapLinearHalf,
        NULL,
        NULL,
        (void*)VirtualTextureGradKernelLinearMipmapNearestBOHalf,
        (void*)VirtualTextureGradKernelLinearMipmapLinearBOHalf,
    };

    // Function index.
    int func_idx = p.filterMode;
    if (p.enableMip && !has_uv_da)
        func_idx += TEX_MODE_COUNT; // Bias-only variant.
    if (dtype == torch::kHalf)
        func_idx += TEX_MODE_COUNT * 2; // Choose half variant.

    // Launch main gradient kernel.
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func_tbl[func_idx], gridSize, blockSize, args, 0, stream));

    // Launch kernel to pull gradients from mip levels. Don't do this if mip stack was supplied - individual level gradients are already there.
    if (p.enableMip)
    {
        int pixel_num_per_thread_x = p.texWidth / 1024;
        int pixel_num_per_thread_y = p.texHeight / 1024;
        int width = p.texWidth / pixel_num_per_thread_x;
        int height = p.texHeight / pixel_num_per_thread_y;
        dim3 blockSize = getLaunchBlockSize(TEX_GRAD_MAX_MIP_KERNEL_BLOCK_WIDTH, TEX_GRAD_MAX_MIP_KERNEL_BLOCK_HEIGHT, width, height);
        dim3 gridSize  = getLaunchGridSize(blockSize, width, height, p.texDepth);
        int sharedBytes = blockSize.x * blockSize.y * p.channels * sizeof(float);
        void* args[] = {&p, &pixel_num_per_thread_x, &pixel_num_per_thread_y};
        void* mip_grad_func_tbl[6] = { 
            (void*)VirtualTextureMipGradKernel1, 
            (void*)VirtualTextureMipGradKernel2, 
            (void*)VirtualTextureMipGradKernel4,
            (void*)VirtualTextureMipGradKernelHalf1, 
            (void*)VirtualTextureMipGradKernelHalf2, 
            (void*)VirtualTextureMipGradKernelHalf4, };
        if (dtype == torch::kHalf)
            channel_div_idx += 3;   // Choose half variant.
        NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(mip_grad_func_tbl[channel_div_idx], gridSize, blockSize, args, sharedBytes, stream));
    }

    // Return output tensors.
    return std::tuple<std::vector<std::vector<torch::Tensor>>, torch::Tensor, torch::Tensor, torch::Tensor >(grad_tex, grad_uv, grad_uv_da, grad_mip_level_bias);
}

// Version for nearest filter mode.
std::vector<torch::Tensor> virtual_texture_grad_nearest(torch::Tensor uv, torch::Tensor dy, torch::Tensor mask, int filter_mode, int boundary_mode, int texture_depth, int texture_height, int texture_width, int texture_channels, int page_size_x, int page_size_y, std::vector<torch::Tensor> pages)
{
    torch::Tensor empty_tensor;
    std::vector<torch::Tensor> empty_vector;
    auto result = virtual_texture_grad_linear_mipmap_linear(uv, dy, empty_tensor, empty_tensor, mask, filter_mode, boundary_mode, texture_depth, texture_height, texture_width, texture_channels, page_size_x, page_size_y, {pages});
    return std::get<0>(result)[0];
}

// Version for linear filter mode.
std::tuple<std::vector<torch::Tensor>, torch::Tensor> virtual_texture_grad_linear(torch::Tensor uv, torch::Tensor dy, torch::Tensor mask, int filter_mode, int boundary_mode, int texture_depth, int texture_height, int texture_width, int texture_channels, int page_size_x, int page_size_y, std::vector<torch::Tensor> pages)
{
    torch::Tensor empty_tensor;
    std::vector<torch::Tensor> empty_vector;
    auto result = virtual_texture_grad_linear_mipmap_linear(uv, dy, empty_tensor, empty_tensor, mask, filter_mode, boundary_mode, texture_depth, texture_height, texture_width, texture_channels, page_size_x, page_size_y, {pages});
    return std::tuple<std::vector<torch::Tensor>, torch::Tensor>(std::get<0>(result)[0], std::get<1>(result));
}

// Version for linear-mipmap-nearest mode.
std::tuple<std::vector<std::vector<torch::Tensor>>, torch::Tensor > virtual_texture_grad_linear_mipmap_nearest(torch::Tensor uv, torch::Tensor dy, torch::Tensor uv_da, torch::Tensor mip_level_bias, torch::Tensor mask, int filter_mode, int boundary_mode, int texture_depth, int texture_height, int texture_width, int texture_channels, int page_size_x, int page_size_y, std::vector<std::vector<torch::Tensor>> pages)
{
    auto result = virtual_texture_grad_linear_mipmap_linear(uv, dy, uv_da, mip_level_bias, mask, filter_mode, boundary_mode, texture_depth, texture_height, texture_width, texture_channels, page_size_x, page_size_y, pages);
    return std::tuple<std::vector<std::vector<torch::Tensor>>, torch::Tensor >(std::get<0>(result), std::get<1>(result));
}

//------------------------------------------------------------------------
// Mipmap op
std::vector<std::vector<torch::Tensor>> virtual_texture_construct_mip_cuda(int max_mip_level, int texture_depth, int texture_height, int texture_width, int texture_channels, int page_size_x, int page_size_y, std::vector<torch::Tensor> pages)
{
    NVDR_CHECK(texture_height>0 && (texture_height & (texture_height-1))==0, "virtual_texture_construct_mip: Texture height must be power of two.");
    NVDR_CHECK(texture_width>0 && (texture_width & (texture_width-1))==0, "virtual_texture_construct_mip: Texture width must be power of two.");
    NVDR_CHECK(page_size_y>0 && (page_size_y & (page_size_y-1))==0, "virtual_texture_construct_mip: Page Y must be power of two.");
    NVDR_CHECK(page_size_x>0 && (page_size_x & (page_size_x-1))==0, "virtual_texture_construct_mip: Page X must be power of two.");

    for (int i = 0; i < pages.size(); i++)
    {
        NVDR_CHECK(pages[i].sizes().size() == 4 && 
                   pages[i].size(0) == texture_depth && 
                   pages[i].size(1) == page_size_y && 
                   pages[i].size(2) == page_size_x && 
                   pages[i].size(3) == texture_channels, "pages[i] must have shape[texture_depth, page_size_y, page_size_x, texture_channels]");
    }
    std::vector<torch::Tensor> cuda_pages;
    for (auto& tensor : pages)
        cuda_pages.push_back(tensor.to(c10::kCUDA, true));
    max_mip_level = calculateMaxMipLevel(texture_width, texture_height, max_mip_level);

    NVDR_CHECK(!cuda_pages.empty(), "virtual_texture_construct_mip: Pages is empty");

    torch::ScalarType dtype = cuda_pages[0].dtype().toScalarType();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(cuda_pages[0]));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    VirtualTextureMipmapParams p = {};
    
    std::vector<std::vector<torch::Tensor>> Out;
    std::vector<torch::Tensor> last_mipmap_pages = cuda_pages;
    for (int mip = 1; mip <= max_mip_level; mip++)
    {
        int2 sz_in = calcMipLevelSize(texture_width, texture_height, mip-1);
        int width_in = sz_in.x;
        int height_in = sz_in.y;
        int2 sz_out = calcMipLevelSize(texture_width, texture_height, mip);
        int width_out = sz_out.x;
        int height_out = sz_out.y;
        int page_num_y_out = calcPageNum(height_out, page_size_y);
        int page_num_x_out = calcPageNum(width_out, page_size_x);
        int page_num_out = page_num_y_out * page_num_x_out;

        std::vector<torch::Tensor> out_pages;
        torch::TensorOptions opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
        for (int i = 0; i < page_num_out; i++)
        {
            int page_width = width_out < page_size_x ? width_out : page_size_x;
            int page_height = height_out < page_size_y ? height_out : page_size_y;
            torch::Tensor mipmap = torch::zeros({texture_depth, page_height, page_width, texture_channels}, opts);
            out_pages.push_back(mipmap);
        }

        auto p_tex = prepareCudaTensorArray(last_mipmap_pages);
        auto p_out = prepareCudaTensorArray<false>(out_pages);
        p.tex = (const float**)p_tex.data_ptr();
        p.out = (float**)p_out.data_ptr();

        p.channels = texture_channels;
        p.texWidth = width_in;
        p.texHeight = height_in;
        p.texDepth = texture_depth;
        p.page_size_x = page_size_x;
        p.page_size_y = page_size_y;

        void* args[] = {&p};
        dim3 blockSize = getLaunchBlockSize(TEX_FWD_MAX_MIP_KERNEL_BLOCK_WIDTH, TEX_FWD_MAX_MIP_KERNEL_BLOCK_HEIGHT, width_out, height_out);
        dim3 gridSize  = getLaunchGridSize(blockSize, width_out, height_out, texture_depth);

        // Choose kernel variants based on channel count.
        int channel_div_idx = 0;
        if (!(p.channels & 3))
            channel_div_idx = 2;  // Channel count divisible by 4.
        else if (!(p.channels & 1))
            channel_div_idx = 1;  // Channel count divisible by 2.

        void* func_tbl[3 * 2] = {
            (void*)VirtualTextureMipmapKernel1,
            (void*)VirtualTextureMipmapKernel2,
            (void*)VirtualTextureMipmapKernel4,
            (void*)VirtualTextureMipmapKernelHalf1,
            (void*)VirtualTextureMipmapKernelHalf2,
            (void*)VirtualTextureMipmapKernelHalf4,
        };
        int func_idx = channel_div_idx;
        if (dtype == torch::kHalf)
            func_idx += 3; // Choose half variant.

        NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func_tbl[func_idx], gridSize, blockSize, args, 0, stream));
    
        last_mipmap_pages = out_pages;
        Out.push_back(out_pages);
    }

    return Out;

}

static __forceinline__ int2 indexPage_CPU(int px, int py, int width, int height, int page_size_x, int page_size_y)
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
static void VirtualTextureMipmapTemplate_CPU(const VirtualTextureMipmapParams& p, int px, int py)
{
    int out_width = p.texWidth / 2;     // Width of outgoing mipmap
    int out_height = p.texHeight / 2;   // Height of outgoing mipmap
    if (px >= out_width || py >= out_height)
        return;

    int2 pi_tc = indexPage_CPU(px, py, out_width, out_height, p.page_size_x, p.page_size_y);
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

            int2 pi_tc_in = indexPage_CPU(px_in, py_in, p.texWidth, p.texHeight, p.page_size_x, p.page_size_y);
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
// Template specializations.
void VirtualTextureMipmapKernel1_CPU                    (const VirtualTextureMipmapParams& p, int px, int py) { VirtualTextureMipmapTemplate_CPU<float,  1>(p, px, py); }
void VirtualTextureMipmapKernel2_CPU                    (const VirtualTextureMipmapParams& p, int px, int py) { VirtualTextureMipmapTemplate_CPU<float2,  2>(p, px, py); }
void VirtualTextureMipmapKernel4_CPU                    (const VirtualTextureMipmapParams& p, int px, int py) { VirtualTextureMipmapTemplate_CPU<float4,  4>(p, px, py); }

std::vector<std::vector<torch::Tensor>> virtual_texture_construct_mip(int max_mip_level, int texture_depth, int texture_height, int texture_width, int texture_channels, int page_size_x, int page_size_y, std::vector<torch::Tensor> pages)
{
    NVDR_CHECK(texture_height>0 && (texture_height & (texture_height-1))==0, "virtual_texture_construct_mip: Texture height must be power of two.");
    NVDR_CHECK(texture_width>0 && (texture_width & (texture_width-1))==0, "virtual_texture_construct_mip: Texture width must be power of two.");
    NVDR_CHECK(page_size_y>0 && (page_size_y & (page_size_y-1))==0, "virtual_texture_construct_mip: Page Y must be power of two.");
    NVDR_CHECK(page_size_x>0 && (page_size_x & (page_size_x-1))==0, "virtual_texture_construct_mip: Page X must be power of two.");

    for (int i = 0; i < pages.size(); i++)
    {
        NVDR_CHECK(pages[i].sizes().size() == 4 && 
                   pages[i].size(0) == texture_depth && 
                   pages[i].size(1) == page_size_y && 
                   pages[i].size(2) == page_size_x && 
                   pages[i].size(3) == texture_channels, "pages[i] must have shape[texture_depth, page_size_y, page_size_x, texture_channels]");
    }

    NVDR_CHECK_F32(pages);
    VirtualTextureMipmapParams p = {};

    max_mip_level = calculateMaxMipLevel(texture_width, texture_height, max_mip_level);

    int thread_num = 1;
    if (std::thread::hardware_concurrency() > 0)
        thread_num = std::thread::hardware_concurrency();

    std::vector<std::vector<torch::Tensor>> Out;
    std::vector<torch::Tensor> last_mipmap_pages = pages;
    for (int mip = 1; mip <= max_mip_level; mip++)
    {
        int2 sz_in = calcMipLevelSize(texture_width, texture_height, mip-1);
        int width_in = sz_in.x;
        int height_in = sz_in.y;
        int2 sz_out = calcMipLevelSize(texture_width, texture_height, mip);
        int width_out = sz_out.x;
        int height_out = sz_out.y;
        int page_num_y_out = calcPageNum(height_out, page_size_y);
        int page_num_x_out = calcPageNum(width_out, page_size_x);
        int page_num_out = page_num_y_out * page_num_x_out;

        std::vector<torch::Tensor> out_pages;
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        int page_width = width_out < page_size_x ? width_out : page_size_x;
        int page_height = height_out < page_size_y ? height_out : page_size_y;
        for (int i = 0; i < page_num_out; i++)
        {
            torch::Tensor mipmap = torch::zeros({texture_depth, page_height, page_width, texture_channels}, opts);
            out_pages.push_back(mipmap);
        }

        std::vector<void*> p_tex;
        for (auto tensor : last_mipmap_pages)
            p_tex.push_back(tensor.data_ptr());
        std::vector<void*> p_out;
        for (auto tensor : out_pages)
            p_out.push_back(tensor.data_ptr());
        p.tex = (const float**)p_tex.data();
        p.out = (float**)p_out.data();

        p.channels = texture_channels;
        p.texWidth = width_in;
        p.texHeight = height_in;
        p.texDepth = texture_depth;
        p.page_size_x = page_size_x;
        p.page_size_y = page_size_y;

        int channel_div_idx = 0;
        if (!(p.channels & 3))
            channel_div_idx = 2;  // Channel count divisible by 4.
        else if (!(p.channels & 1))
            channel_div_idx = 1;  // Channel count divisible by 2.

        std::function<void(const VirtualTextureMipmapParams, int, int)> func_tbl[3] = {
            VirtualTextureMipmapKernel1_CPU,
            VirtualTextureMipmapKernel2_CPU,
            VirtualTextureMipmapKernel4_CPU,
        };
        int func_idx = channel_div_idx;
        auto Function = func_tbl[func_idx];

        int num_rows_per_thread = height_out / thread_num;
        ParallelFor(thread_num, thread_num, [&](int thread_index) {
            int row_start = thread_index * num_rows_per_thread;
            int row_end = (thread_index + 1) * num_rows_per_thread;
            if (thread_index == thread_num - 1)
                row_end = height_out;
            for (int py = row_start; py < row_end; py++)
                for (int px = 0; px < width_out; px++)
                    Function(p, px, py);
        });
    
        last_mipmap_pages = out_pages;
        Out.push_back(out_pages);
    }

    return Out;
}