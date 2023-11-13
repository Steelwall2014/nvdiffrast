
#include "torch_common.inl"
#include "torch_types.h"
#include "../common/common.h"
#include "../common/virtual_texture.h"
#include <cuda_runtime.h>
#include <torch/extension.h>

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
void VirtualTextureGradKernelNearest                        (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelLinear                         (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelLinearMipmapNearest            (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelLinearMipmapLinear             (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelLinearMipmapNearestBO          (const VirtualTextureKernelParams p);
void VirtualTextureGradKernelLinearMipmapLinearBO           (const VirtualTextureKernelParams p);
void VirtualTextureMipmapKernel1                            (const VirtualTextureMipmapParams p);
void VirtualTextureMipmapKernel2                            (const VirtualTextureMipmapParams p);
void VirtualTextureMipmapKernel4                            (const VirtualTextureMipmapParams p);
void VirtualTextureMipGradKernel1                           (const VirtualTextureKernelParams p, int, int);
void VirtualTextureMipGradKernel2                           (const VirtualTextureKernelParams p, int, int);
void VirtualTextureMipGradKernel4                           (const VirtualTextureKernelParams p, int, int);

int calculateMaxMipLevel(int width, int height, int mipLevelLimit)
{
    // If max_mip_level is equal to -1, it means we should decide maximum mipmap level by ourselves.
    // Then make max_mip_level big enough to ignore this value.
    // if (max_mip_level == -1)
    //     max_mip_level = 0x7fffffff;
    // max_mip_level = std::min((int)std::log2(std::min(height, width)), max_mip_level);
    // return max_mip_level;

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

std::vector<torch::Tensor> virtual_texture_feedback_mip(torch::Tensor uv, torch::Tensor uv_da, torch::Tensor mip_level_bias, int filter_mode, int boundary_mode, int texture_depth, int texture_height, int texture_width, int texture_channels, int page_size_x, int page_size_y, std::vector<std::vector<torch::Tensor>> pages)
{
    NVDR_CHECK(texture_height>0 && (texture_height & (texture_height-1))==0, "virtual_texture_feedback_mip: Texture height must be power of two.");
    NVDR_CHECK(texture_width>0 && (texture_width & (texture_width-1))==0, "virtual_texture_feedback_mip: Texture width must be power of two.");
    NVDR_CHECK(page_size_y>0 && (page_size_y & (page_size_y-1))==0, "virtual_texture_feedback_mip: Page Y must be power of two.");
    NVDR_CHECK(page_size_x>0 && (page_size_x & (page_size_x-1))==0, "virtual_texture_feedback_mip: Page X must be power of two.");
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(uv));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    VirtualTextureKernelParams p = {}; // Initialize all fields to zero.
    int max_mip_level = pages.size()-1;//calculateMaxMipLevel(texture_width, texture_height, -1);
    set_modes(p, filter_mode, boundary_mode, max_mip_level);
    p.mipLevelMax = max_mip_level;

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
        // int2 sz = calcMipLevelSize(texture_width, texture_height, mip);
        // int page_num_y = calcPageNum(sz.y, page_size_y);
        // int page_num_x = calcPageNum(sz.x, page_size_x);
        int numPages = pages[mip].size();
        // Allocate output tensor.
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
        torch::Tensor feedback = torch::zeros({numPages}, opts);
        Out[mip] = feedback;
        p.feedback[mip] = feedback.data_ptr<unsigned char>();
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

    for (int mip = 0; mip <= max_mip_level; mip++)
    {
        int numPages = pages[mip].size();
        for (int i = 0; i < numPages; i++)
        {
            if (Out[mip][i].item<unsigned char>() == 1)
                pages[mip][i].requires_grad_(true);
            else
                pages[mip][i].requires_grad_(false);
        }
    }

    // Return output tensor.
    return Out;

}

// Version without mipmaps.
std::vector<torch::Tensor> virtual_texture_feedback(torch::Tensor uv, int filter_mode, int boundary_mode, int texture_depth, int texture_height, int texture_width, int texture_channels, int page_size_x, int page_size_y, std::vector<std::vector<torch::Tensor>> pages)
{
    torch::Tensor empty_tensor;
    return virtual_texture_feedback_mip(uv, empty_tensor, empty_tensor, filter_mode, boundary_mode, texture_depth, texture_height, texture_width, texture_channels, page_size_x, page_size_y, pages);
}

//------------------------------------------------------------------------
// Forward op.

void* prepareCudaPages(const std::vector<torch::Tensor>& pages)
{
    float** cuda_pages;
    cudaMalloc(&cuda_pages, sizeof(float*) * pages.size());
    float** temp = new float*[pages.size()];
    for (int page_index = 0; page_index < pages.size(); page_index++)
    {
        if (pages[page_index].defined() && pages[page_index].nbytes())
        {
            temp[page_index] = pages[page_index].data_ptr<float>();
        }
        else
        {
            temp[page_index] = NULL;
        }
    }
    cudaMemcpy(cuda_pages, temp, sizeof(float*) * pages.size(), cudaMemcpyHostToDevice);
    delete[] temp;
    return cuda_pages;
}

torch::Tensor virtual_texture_fwd_mip(
    torch::Tensor uv, torch::Tensor uv_da, torch::Tensor mip_level_bias, 
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
    for (auto& mip_pages : pages)
    {
        NVDR_CHECK_DEVICE(mip_pages);
        NVDR_CHECK_CONTIGUOUS(mip_pages);
        NVDR_CHECK_F32(mip_pages);
        // for (auto& mip_page : mip_pages)
        // {
        //     NVDR_CHECK(mip_page.sizes().size() == 4 && 
        //                mip_page.size(0) > 0 && 
        //                mip_page.size(1) == page_size_y && 
        //                mip_page.size(2) == page_size_x && 
        //                mip_page.size(3) == texture_channels, 
        //                "virtual texture pages must have shape[>0, page_size_y, page_size_x, texture_channels]");
        // }

    }
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


    for (int mip = 0; mip <= max_mip_level; mip++)
    {
        int2 sz = calcMipLevelSize(texture_width, texture_height, mip);
        int page_num_y = calcPageNum(sz.y, page_size_y);
        int page_num_x = calcPageNum(sz.x, page_size_x);
        int numPages = page_num_x*page_num_y;
        NVDR_CHECK(numPages == pages[mip].size(), "virtual_texture_fwd_mip: The number of pages mismatches in mipmap level " + std::to_string(mip));
        p.tex[mip] = (const float**)prepareCudaPages(pages[mip]);
    }
    p.uv = uv.data_ptr<float>();
    p.uvDA = (p.enableMip && has_uv_da) ? uv_da.data_ptr<float>() : NULL;
    p.mipLevelBias = (p.enableMip && has_mip_level_bias) ? mip_level_bias.data_ptr<float>() : NULL;

    // Allocate output tensor.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({p.n, p.imgHeight, p.imgWidth, p.channels}, opts);
    p.out = out.data_ptr<float>();

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
    void* func_tbl[TEX_MODE_COUNT * 2 * 3] = {
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
    };

    // Function index.
    int func_idx = p.filterMode;
    if (p.enableMip && !has_uv_da)
        func_idx += TEX_MODE_COUNT; // Bias-only variant.
    func_idx = func_idx * 3 + channel_div_idx; // Choose vector size.

    // Launch kernel.
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func_tbl[func_idx], gridSize, blockSize, args, 0, stream));

    for (int mip = 0; mip <= max_mip_level; mip++)
        cudaFree(p.tex[mip]);
    // Return output tensor.
    return out;
}

// Version without mipmaps.
torch::Tensor virtual_texture_fwd(
    torch::Tensor uv, 
    int filter_mode, int boundary_mode,
    int texture_depth, int texture_height, int texture_width, int texture_channels,
    int page_size_x, int page_size_y, 
    std::vector<torch::Tensor> pages)
{
    torch::Tensor empty_tensor;
    std::vector<torch::Tensor> empty_vector;
    return virtual_texture_fwd_mip(
        uv, empty_tensor, empty_tensor, 
        filter_mode, boundary_mode,
        texture_depth, texture_height, texture_width, texture_channels,
        page_size_x, page_size_y, 
        {pages});
}

//------------------------------------------------------------------------
// Gradient op.

std::tuple<std::vector<std::vector<torch::Tensor>>, torch::Tensor, torch::Tensor, torch::Tensor > 
virtual_texture_grad_linear_mipmap_linear(torch::Tensor uv, torch::Tensor dy, torch::Tensor uv_da, torch::Tensor mip_level_bias, int filter_mode, int boundary_mode, int texture_depth, int texture_height, int texture_width, int texture_channels, int page_size_x, int page_size_y, std::vector<std::vector<torch::Tensor>> pages)
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
    for (auto& mip_pages : pages)
    {
        NVDR_CHECK_DEVICE(mip_pages);
        NVDR_CHECK_CONTIGUOUS(mip_pages);
        NVDR_CHECK_F32(mip_pages);
        // for (auto& mip_page : mip_pages)
        // {
        //     NVDR_CHECK(mip_page.sizes().size() == 4 && 
        //                mip_page.size(0) > 0 && 
        //                mip_page.size(1) == page_size_y && 
        //                mip_page.size(2) == page_size_x && 
        //                mip_page.size(3) == texture_channels, 
        //                "virtual texture pages must have shape[>0, page_size_y, page_size_x, texture_channels]");
        // }

    }
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

    for (int mip=0; mip <= max_mip_level; mip++)
    {
        p.tex[mip] = (const float**)prepareCudaPages(pages[mip]);
    }
    p.uv = uv.data_ptr<float>();
    p.dy = dy_.data_ptr<float>();
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
            if (pages[mip][page_index].defined() && pages[mip][page_index].nbytes())
            {
                torch::Tensor grad_page = torch::zeros_like(pages[mip][page_index]);
                mip_grad_tex[page_index] = grad_page;
            }
        }
        p.gradTex[mip] = (float**)prepareCudaPages(mip_grad_tex);
        grad_tex.push_back(mip_grad_tex);
    }

    // Allocate output tensor for uv gradient.
    torch::Tensor grad_uv;
    torch::Tensor grad_uv_da;
    torch::Tensor grad_mip_level_bias;
    if (p.filterMode != TEX_MODE_NEAREST)
    {
        grad_uv = torch::empty_like(uv);
        p.gradUV = grad_uv.data_ptr<float>();

        // Gradients for things affecting mip level.
        if (p.filterMode == TEX_MODE_LINEAR_MIPMAP_LINEAR)
        {
            // Allocate output tensor for uv_da gradient.
            if (has_uv_da)
            {
                grad_uv_da = torch::empty_like(uv_da);
                p.gradUVDA = grad_uv_da.data_ptr<float>();
            }

            // Allocate output tensor for mip_level_bias gradient.
            if (has_mip_level_bias)
            {
                grad_mip_level_bias = torch::empty_like(mip_level_bias);
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

    void* func_tbl[TEX_MODE_COUNT * 2] = {
        (void*)VirtualTextureGradKernelNearest,
        (void*)VirtualTextureGradKernelLinear,
        (void*)VirtualTextureGradKernelLinearMipmapNearest,
        (void*)VirtualTextureGradKernelLinearMipmapLinear,
        NULL,
        NULL,
        (void*)VirtualTextureGradKernelLinearMipmapNearestBO,
        (void*)VirtualTextureGradKernelLinearMipmapLinearBO,
    };

    // Function index.
    int func_idx = p.filterMode;
    if (p.enableMip && !has_uv_da)
        func_idx += TEX_MODE_COUNT; // Bias-only variant.

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
        void* mip_grad_func_tbl[3] = { (void*)VirtualTextureMipGradKernel1, (void*)VirtualTextureMipGradKernel2, (void*)VirtualTextureMipGradKernel4 };
        NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(mip_grad_func_tbl[channel_div_idx], gridSize, blockSize, args, sharedBytes, stream));
    }

    for (int mip = 0; mip <= max_mip_level; mip++)
        cudaFree(p.tex[mip]);
    // Return output tensors.
    return std::tuple<std::vector<std::vector<torch::Tensor>>, torch::Tensor, torch::Tensor, torch::Tensor >(grad_tex, grad_uv, grad_uv_da, grad_mip_level_bias);
}

// Version for nearest filter mode.
std::vector<torch::Tensor> virtual_texture_grad_nearest(torch::Tensor uv, torch::Tensor dy, int filter_mode, int boundary_mode, int texture_depth, int texture_height, int texture_width, int texture_channels, int page_size_x, int page_size_y, std::vector<torch::Tensor> pages)
{
    torch::Tensor empty_tensor;
    std::vector<torch::Tensor> empty_vector;
    auto result = virtual_texture_grad_linear_mipmap_linear(uv, dy, empty_tensor, empty_tensor, filter_mode, boundary_mode, texture_depth, texture_height, texture_width, texture_channels, page_size_x, page_size_y, {pages});
    return std::get<0>(result)[0];
}

// Version for linear filter mode.
std::tuple<std::vector<torch::Tensor>, torch::Tensor> virtual_texture_grad_linear(torch::Tensor uv, torch::Tensor dy, int filter_mode, int boundary_mode, int texture_depth, int texture_height, int texture_width, int texture_channels, int page_size_x, int page_size_y, std::vector<torch::Tensor> pages)
{
    torch::Tensor empty_tensor;
    std::vector<torch::Tensor> empty_vector;
    auto result = virtual_texture_grad_linear_mipmap_linear(uv, dy, empty_tensor, empty_tensor, filter_mode, boundary_mode, texture_depth, texture_height, texture_width, texture_channels, page_size_x, page_size_y, {pages});
    return std::tuple<std::vector<torch::Tensor>, torch::Tensor>(std::get<0>(result)[0], std::get<1>(result));
}

// Version for linear-mipmap-nearest mode.
std::tuple<std::vector<std::vector<torch::Tensor>>, torch::Tensor > virtual_texture_grad_linear_mipmap_nearest(torch::Tensor uv, torch::Tensor dy, torch::Tensor uv_da, torch::Tensor mip_level_bias, int filter_mode, int boundary_mode, int texture_depth, int texture_height, int texture_width, int texture_channels, int page_size_x, int page_size_y, std::vector<std::vector<torch::Tensor>> pages)
{
    auto result = virtual_texture_grad_linear_mipmap_linear(uv, dy, uv_da, mip_level_bias, filter_mode, boundary_mode, texture_depth, texture_height, texture_width, texture_channels, page_size_x, page_size_y, pages);
    return std::tuple<std::vector<std::vector<torch::Tensor>>, torch::Tensor >(std::get<0>(result), std::get<1>(result));
}

//------------------------------------------------------------------------
// Mipmap op
std::vector<std::vector<torch::Tensor>> virtual_texture_construct_mip(int max_mip_level, int texture_depth, int texture_height, int texture_width, int texture_channels, int page_size_x, int page_size_y, std::vector<torch::Tensor> pages)
{
    NVDR_CHECK(!pages.empty(), "virtual_texture_construct_mip: Pages is empty");
    NVDR_CHECK_DEVICE(pages);
    NVDR_CHECK_CONTIGUOUS(pages);
    NVDR_CHECK_F32(pages);
    NVDR_CHECK(texture_height>0 && (texture_height & (texture_height-1))==0, "virtual_texture_construct_mip: Texture height must be power of two.");
    NVDR_CHECK(texture_width>0 && (texture_width & (texture_width-1))==0, "virtual_texture_construct_mip: Texture width must be power of two.");
    NVDR_CHECK(page_size_y>0 && (page_size_y & (page_size_y-1))==0, "virtual_texture_construct_mip: Page Y must be power of two.");
    NVDR_CHECK(page_size_x>0 && (page_size_x & (page_size_x-1))==0, "virtual_texture_construct_mip: Page X must be power of two.");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(pages[0]));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    for (int i = 0; i < pages.size(); i++)
    {
        NVDR_CHECK(pages[i].sizes().size() == 4 && 
                   pages[i].size(0) == texture_depth && 
                   pages[i].size(1) == page_size_y && 
                   pages[i].size(2) == page_size_x && 
                   pages[i].size(3) == texture_channels, "pages[i] must have shape[texture_depth, page_size_y, page_size_x, texture_channels]");
    }

    max_mip_level = calculateMaxMipLevel(texture_width, texture_height, max_mip_level);

    VirtualTextureMipmapParams p = {};
    
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
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        for (int i = 0; i < page_num_out; i++)
        {
            int page_width = width_out < page_size_x ? width_out : page_size_x;
            int page_height = height_out < page_size_y ? height_out : page_size_y;
            torch::Tensor mipmap = torch::zeros({texture_depth, page_height, page_width, texture_channels}, opts);
            out_pages.push_back(mipmap);
        }
        // if (height_out < page_size_y || width_out < page_size_x)
        // {
        //     // If the outgoing mipmap is smaller than the size of a page, 
        //     // then this mipmap level will be stored at its original size.
        //     torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        //     torch::Tensor mipmap = torch::zeros({texture_depth, height_out, width_out, texture_channels}, opts);
        //     out_pages.push_back(mipmap);
        // }
        // else
        // {
        //     // Since both height_out/width_out and page_size_y/page_size_x are power of two,
        //     // page_num_y_out and page_num_x_out will be integers.
        //     int page_num_y_out = height_out / page_size_y;
        //     int page_num_x_out = width_out / page_size_x;
        //     int page_num_out = page_num_y_out * page_num_x_out;
        //     torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        //     for (int i = 0; i < page_num_out; i++)
        //     {
        //         torch::Tensor mipmap = torch::zeros({texture_depth, page_size_y, page_size_x, texture_channels}, opts);
        //         out_pages.push_back(mipmap);
        //     }
        // }

        p.tex = (float const**)prepareCudaPages(last_mipmap_pages);
        p.out = (float**)prepareCudaPages(out_pages);

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

        void* func_tbl[3] = {
            (void*)VirtualTextureMipmapKernel1,
            (void*)VirtualTextureMipmapKernel2,
            (void*)VirtualTextureMipmapKernel4,
        };
        int func_idx = channel_div_idx;

        NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func_tbl[func_idx], gridSize, blockSize, args, 0, stream));
    
        cudaFree(p.tex);
        cudaFree(p.out);
        last_mipmap_pages = out_pages;
        Out.push_back(out_pages);
    }

    return Out;

}