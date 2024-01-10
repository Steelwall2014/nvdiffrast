#include "../common/common.h"
#include "../common/rasterize.h"
#include "virtual_geometry.h"


static __device__ __forceinline__ bool IsOutSidePlane(const Plane &plane, float x, float y, float z)
{
    return plane.Normal[0]*x + plane.Normal[1]*y + plane.Normal[2]*z + plane.Distance < 0;
}
static __device__ __forceinline__ bool IsBoxOutSidePlane(const Plane &plane, const BoundingBox &AABB)
{
    return 
        IsOutSidePlane(plane, AABB.XMin, AABB.YMin, AABB.ZMin) && 
        IsOutSidePlane(plane, AABB.XMin, AABB.YMin, AABB.ZMax) &&
        IsOutSidePlane(plane, AABB.XMin, AABB.YMax, AABB.ZMin) &&
        IsOutSidePlane(plane, AABB.XMin, AABB.YMax, AABB.ZMax) &&
        IsOutSidePlane(plane, AABB.XMax, AABB.YMin, AABB.ZMin) &&
        IsOutSidePlane(plane, AABB.XMax, AABB.YMin, AABB.ZMax) &&
        IsOutSidePlane(plane, AABB.XMax, AABB.YMax, AABB.ZMin) &&
        IsOutSidePlane(plane, AABB.XMax, AABB.YMax, AABB.ZMax);
}

static __device__ __forceinline__ bool IsBoxOutSideFrustum(const ViewFrustum& Frustum, const BoundingBox &AABB)
{
    return 
        IsBoxOutSidePlane(Frustum.Planes[0], AABB) || 
        IsBoxOutSidePlane(Frustum.Planes[1], AABB) || 
        IsBoxOutSidePlane(Frustum.Planes[2], AABB) || 
        IsBoxOutSidePlane(Frustum.Planes[3], AABB) || 
        IsBoxOutSidePlane(Frustum.Planes[4], AABB) || 
        IsBoxOutSidePlane(Frustum.Planes[5], AABB);
}

__global__ void VirtualGeometryFrustumCullKernal(VirtualGeometryFrustumCullParams p)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z; 
    int cluster_index = px + py*blockDim.x*gridDim.x;
    if (cluster_index >= p.numClusters)
        return;

    const ViewFrustum& frustum = p.frustums[pz];
    // const BoundingBox& AABB = p.AABBs[cluster_index];
    BoundingBox AABB = p.AABBs[cluster_index];

    bool culled = IsBoxOutSideFrustum(frustum, AABB);

    bool* pCulled = p.culled + p.numClusters*pz + cluster_index;
    *pCulled = culled;
}

//------------------------------------------------------------------------
// Cuda forward rasterizer pixel shader kernel.

__global__ void VirtualGeometryRasterizeCudaFwdShaderKernel(const VirtualGeometryRasterizeCudaFwdShaderParams p)
{
    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)
        return;

    // Pixel index.
    int pidx = px + p.width * (py + p.height * pz);

    // Fetch triangle idx.
    // union
    // {
    //     uint32_t idx;
    //     float out_w;
    //     struct
    //     {
    //         uint16_t clusterIndex;
    //         uint16_t triIndex;
    //     };
    // } index_union;
    // index_union.idx = ((const uint32_t*)p.in_idx)[pidx];
    // int triIdx = index_union.triIndex - 1;
    // int clusterIndex = index_union.clusterIndex;
    // float out_w = index_union.out_w;

    int triIdx = p.in_idx[pidx] - 1;

    if (triIdx < 0 || triIdx >= p.numTriangles*p.numClusters)
    {
        // No or corrupt triangle.
        ((float4*)p.out)[pidx] = make_float4(0.0, 0.0, 0.0, 0.0); // Clear out.
        ((float4*)p.out_db)[pidx] = make_float4(0.0, 0.0, 0.0, 0.0); // Clear out_db.
        return;
    }

    int clusterIndex = triIdx / p.numTriangles;
    triIdx = triIdx % p.numTriangles;

    // Fetch vertex indices.
    int vi0 = p.tri[clusterIndex][triIdx * 3 + 0];
    int vi1 = p.tri[clusterIndex][triIdx * 3 + 1];
    int vi2 = p.tri[clusterIndex][triIdx * 3 + 2];

    // Bail out if vertex indices are corrupt.
    if (vi0 < 0 || vi0 >= p.numVertices[clusterIndex] ||
        vi1 < 0 || vi1 >= p.numVertices[clusterIndex] ||
        vi2 < 0 || vi2 >= p.numVertices[clusterIndex])
        return;

    // In instance mode, adjust vertex indices by minibatch index.
    if (p.instance_mode)
    {
        vi0 += pz * p.numVertices[clusterIndex];
        vi1 += pz * p.numVertices[clusterIndex];
        vi2 += pz * p.numVertices[clusterIndex];
    }

    // Fetch vertex positions.
    float4 p0 = ((float4*)p.pos[clusterIndex])[vi0];
    float4 p1 = ((float4*)p.pos[clusterIndex])[vi1];
    float4 p2 = ((float4*)p.pos[clusterIndex])[vi2];

    // Evaluate edge functions.
    float fx = p.xs * (float)px + p.xo;
    float fy = p.ys * (float)py + p.yo;
    float p0x = p0.x - fx * p0.w;
    float p0y = p0.y - fy * p0.w;
    float p1x = p1.x - fx * p1.w;
    float p1y = p1.y - fy * p1.w;
    float p2x = p2.x - fx * p2.w;
    float p2y = p2.y - fy * p2.w;
    float a0 = p1x*p2y - p1y*p2x;
    float a1 = p2x*p0y - p2y*p0x;
    float a2 = p0x*p1y - p0y*p1x;

    // Perspective correct, normalized barycentrics.
    float iw = 1.f / (a0 + a1 + a2);
    float b0 = a0 * iw;
    float b1 = a1 * iw;

    // Compute z/w for depth buffer.
    float z = p0.z * a0 + p1.z * a1 + p2.z * a2;
    float w = p0.w * a0 + p1.w * a1 + p2.w * a2;
    float zw = z / w;

    // Clamps to avoid NaNs.
    b0 = __saturatef(b0); // Clamp to [+0.0, 1.0].
    b1 = __saturatef(b1); // Clamp to [+0.0, 1.0].
    zw = fmaxf(fminf(zw, 1.f), -1.f);

    // Emit output.
    ((float4*)p.out)[pidx] = make_float4(b0, b1, zw, (float)p.in_idx[pidx]);

    // Calculate bary pixel differentials.
    float dfxdx = p.xs * iw;
    float dfydy = p.ys * iw;
    float da0dx = p2.y*p1.w - p1.y*p2.w;
    float da0dy = p1.x*p2.w - p2.x*p1.w;
    float da1dx = p0.y*p2.w - p2.y*p0.w;
    float da1dy = p2.x*p0.w - p0.x*p2.w;
    float da2dx = p1.y*p0.w - p0.y*p1.w;
    float da2dy = p0.x*p1.w - p1.x*p0.w;
    float datdx = da0dx + da1dx + da2dx;
    float datdy = da0dy + da1dy + da2dy;
    float dudx = dfxdx * (b0 * datdx - da0dx);
    float dudy = dfydy * (b0 * datdy - da0dy);
    float dvdx = dfxdx * (b1 * datdx - da1dx);
    float dvdy = dfydy * (b1 * datdy - da1dy);

    // Emit bary pixel differentials.
    ((float4*)p.out_db)[pidx] = make_float4(dudx, dudy, dvdx, dvdy);

    p.out_mat[pidx] = p.mat_ids[clusterIndex];
}

//------------------------------------------------------------------------
// Gradient Cuda kernel.

template <bool ENABLE_DB>
static __forceinline__ __device__ void VirtualGeometryRasterizeGradKernelTemplate(const VirtualGeometryRasterizeGradParams p)
{
    // Temporary space for coalesced atomics.
    CA_DECLARE_TEMP(RAST_GRAD_MAX_KERNEL_BLOCK_WIDTH * RAST_GRAD_MAX_KERNEL_BLOCK_HEIGHT);

    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)
        return;

    // Pixel index.
    int pidx = px + p.width * (py + p.height * pz);

    // Read triangle idx and dy.
    float2 dy  = ((float2*)p.dy)[pidx * 2];
    float4 ddb = ENABLE_DB ? ((float4*)p.ddb)[pidx] : make_float4(0.f, 0.f, 0.f, 0.f);
    int triIdx = (int)(((float*)p.out)[pidx * 4 + 3]) - 1;
    int tri_idx = triIdx;
    // union
    // {
    //     uint32_t idx;
    //     float out_w;
    //     struct
    //     {
    //         uint16_t clusterIndex;
    //         uint16_t triIndex;
    //     };
    // } index_union;
    // index_union.out_w = ((float*)p.out)[pidx * 4 + 3];
    // int triIdx = (int)index_union.triIndex - 1;
    // int clusterIndex = index_union.clusterIndex;

    // Exit if nothing to do.
    if (triIdx < 0 || triIdx >= p.numTriangles*p.numClusters)
        return; // No or corrupt triangle.
    int clusterIndex = triIdx / p.numTriangles;
    triIdx = triIdx % p.numTriangles;
    int grad_all_dy = __float_as_int(dy.x) | __float_as_int(dy.y); // Bitwise OR of all incoming gradients.
    int grad_all_ddb = 0;
    if (ENABLE_DB)
        grad_all_ddb = __float_as_int(ddb.x) | __float_as_int(ddb.y) | __float_as_int(ddb.z) | __float_as_int(ddb.w);
    if (((grad_all_dy | grad_all_ddb) << 1) == 0)
        return; // All incoming gradients are +0/-0.

    // Fetch vertex indices.
    int vi0 = p.tri[clusterIndex][triIdx * 3 + 0];
    int vi1 = p.tri[clusterIndex][triIdx * 3 + 1];
    int vi2 = p.tri[clusterIndex][triIdx * 3 + 2];

    // Bail out if vertex indices are corrupt.
    if (vi0 < 0 || vi0 >= p.numVertices[clusterIndex] ||
        vi1 < 0 || vi1 >= p.numVertices[clusterIndex] ||
        vi2 < 0 || vi2 >= p.numVertices[clusterIndex])
        return;

    // In instance mode, adjust vertex indices by minibatch index.
    if (p.instance_mode)
    {
        vi0 += pz * p.numVertices[clusterIndex];
        vi1 += pz * p.numVertices[clusterIndex];
        vi2 += pz * p.numVertices[clusterIndex];
    }

    // Initialize coalesced atomics.
    CA_SET_GROUP(tri_idx);

    // Fetch vertex positions.
    float4 p0 = ((float4*)p.pos[clusterIndex])[vi0];
    float4 p1 = ((float4*)p.pos[clusterIndex])[vi1];
    float4 p2 = ((float4*)p.pos[clusterIndex])[vi2];

    // Evaluate edge functions.
    float fx = p.xs * (float)px + p.xo;
    float fy = p.ys * (float)py + p.yo;
    float p0x = p0.x - fx * p0.w;
    float p0y = p0.y - fy * p0.w;
    float p1x = p1.x - fx * p1.w;
    float p1y = p1.y - fy * p1.w;
    float p2x = p2.x - fx * p2.w;
    float p2y = p2.y - fy * p2.w;
    float a0 = p1x*p2y - p1y*p2x;
    float a1 = p2x*p0y - p2y*p0x;
    float a2 = p0x*p1y - p0y*p1x;

    // Compute inverse area with epsilon.
    float at = a0 + a1 + a2;
    float ep = copysignf(1e-6f, at); // ~1 pixel in 1k x 1k image.
    float iw = 1.f / (at + ep);

    // Perspective correct, normalized barycentrics.
    float b0 = a0 * iw;
    float b1 = a1 * iw;

    // Position gradients.
    float gb0  = dy.x * iw;
    float gb1  = dy.y * iw;
    float gbb  = gb0 * b0 + gb1 * b1;
    float gp0x = gbb * (p2y - p1y) - gb1 * p2y;
    float gp1x = gbb * (p0y - p2y) + gb0 * p2y;
    float gp2x = gbb * (p1y - p0y) - gb0 * p1y + gb1 * p0y;
    float gp0y = gbb * (p1x - p2x) + gb1 * p2x;
    float gp1y = gbb * (p2x - p0x) - gb0 * p2x;
    float gp2y = gbb * (p0x - p1x) + gb0 * p1x - gb1 * p0x;
    float gp0w = -fx * gp0x - fy * gp0y;
    float gp1w = -fx * gp1x - fy * gp1y;
    float gp2w = -fx * gp2x - fy * gp2y;

    // Bary differential gradients.
    if (ENABLE_DB && ((grad_all_ddb) << 1) != 0)
    {
        float dfxdX = p.xs * iw;
        float dfydY = p.ys * iw;
        ddb.x *= dfxdX;
        ddb.y *= dfydY;
        ddb.z *= dfxdX;
        ddb.w *= dfydY;

        float da0dX = p1.y * p2.w - p2.y * p1.w;
        float da1dX = p2.y * p0.w - p0.y * p2.w;
        float da2dX = p0.y * p1.w - p1.y * p0.w;
        float da0dY = p2.x * p1.w - p1.x * p2.w;
        float da1dY = p0.x * p2.w - p2.x * p0.w;
        float da2dY = p1.x * p0.w - p0.x * p1.w;
        float datdX = da0dX + da1dX + da2dX;
        float datdY = da0dY + da1dY + da2dY;

        float x01 = p0.x - p1.x;
        float x12 = p1.x - p2.x;
        float x20 = p2.x - p0.x;
        float y01 = p0.y - p1.y;
        float y12 = p1.y - p2.y;
        float y20 = p2.y - p0.y;
        float w01 = p0.w - p1.w;
        float w12 = p1.w - p2.w;
        float w20 = p2.w - p0.w;

        float a0p1 = fy * p2.x - fx * p2.y;
        float a0p2 = fx * p1.y - fy * p1.x;
        float a1p0 = fx * p2.y - fy * p2.x;
        float a1p2 = fy * p0.x - fx * p0.y;

        float wdudX = 2.f * b0 * datdX - da0dX; 
        float wdudY = 2.f * b0 * datdY - da0dY;
        float wdvdX = 2.f * b1 * datdX - da1dX;
        float wdvdY = 2.f * b1 * datdY - da1dY;

        float c0  = iw * (ddb.x * wdudX + ddb.y * wdudY + ddb.z * wdvdX + ddb.w * wdvdY);
        float cx  = c0 * fx - ddb.x * b0 - ddb.z * b1;
        float cy  = c0 * fy - ddb.y * b0 - ddb.w * b1;
        float cxy = iw * (ddb.x * datdX + ddb.y * datdY);
        float czw = iw * (ddb.z * datdX + ddb.w * datdY);

        gp0x += c0 * y12 - cy * w12              + czw * p2y                                               + ddb.w * p2.w;
        gp1x += c0 * y20 - cy * w20 - cxy * p2y                              - ddb.y * p2.w;
        gp2x += c0 * y01 - cy * w01 + cxy * p1y  - czw * p0y                 + ddb.y * p1.w                - ddb.w * p0.w;
        gp0y += cx * w12 - c0 * x12              - czw * p2x                                - ddb.z * p2.w;
        gp1y += cx * w20 - c0 * x20 + cxy * p2x               + ddb.x * p2.w;
        gp2y += cx * w01 - c0 * x01 - cxy * p1x  + czw * p0x  - ddb.x * p1.w                + ddb.z * p0.w;
        gp0w += cy * x12 - cx * y12              - czw * a1p0                               + ddb.z * p2.y - ddb.w * p2.x;
        gp1w += cy * x20 - cx * y20 - cxy * a0p1              - ddb.x * p2.y + ddb.y * p2.x;
        gp2w += cy * x01 - cx * y01 - cxy * a0p2 - czw * a1p2 + ddb.x * p1.y - ddb.y * p1.x - ddb.z * p0.y + ddb.w * p0.x;
    }

    // Accumulate using coalesced atomics.
    caAtomicAdd3_xyw(p.grad[clusterIndex] + 4 * vi0, gp0x, gp0y, gp0w);
    caAtomicAdd3_xyw(p.grad[clusterIndex] + 4 * vi1, gp1x, gp1y, gp1w);
    caAtomicAdd3_xyw(p.grad[clusterIndex] + 4 * vi2, gp2x, gp2y, gp2w);
}

// Template specializations.
__global__ void VirtualGeometryRasterizeGradKernel  (const VirtualGeometryRasterizeGradParams p) { VirtualGeometryRasterizeGradKernelTemplate<false>(p); }
__global__ void VirtualGeometryRasterizeGradKernelDb(const VirtualGeometryRasterizeGradParams p) { VirtualGeometryRasterizeGradKernelTemplate<true>(p); }

//------------------------------------------------------------------------
// Accumulate gradients kernel.
__global__ void VirtualGeometryAggregateGradKernel(const VirtualGeometryAccumulateGradParams p)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int group_index = px + py*blockDim.x*gridDim.x;
    if (group_index >= p.numGroups)
        return;

    int offset = p.offsetGroups[group_index];
    int* pGroup = p.matchingVerts + offset;
    int numVerts = (p.offsetGroups[group_index+1] - offset) / 2;

    for (int j = 0; j < p.numAttr; j++)
    {
        float grad = 0.f;
        for (int i = 0; i < numVerts; i++)
        {
            int cluster_index = pGroup[i*2];
            int vertex_index = pGroup[i*2+1];
            int ptr = vertex_index * p.numAttr + j;
            if (p.grad[cluster_index])
                grad += p.grad[cluster_index][ptr];
        }

        for (int i = 0; i < numVerts; i++)
        {
            int cluster_index = pGroup[i*2];
            int vertex_index = pGroup[i*2+1];
            int ptr = vertex_index * p.numAttr + j;
            if (p.grad[cluster_index])
                p.grad[cluster_index][ptr] = grad;
        }
    }
    
}