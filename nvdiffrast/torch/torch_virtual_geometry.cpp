#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../common/framework.h"
#include "../common/virtual_geometry_partition.h"
#include "../common/virtual_geometry.h"
#include "../common/parallel.hpp"
#include "torch_types.h"

void VirtualGeometryFrustumCullKernal(VirtualGeometryFrustumCullParams p);
void VirtualGeometryAllReduceKernelSum(VirtualGeometryAllReduceParams p);
void VirtualGeometryAllReduceKernelAvg(VirtualGeometryAllReduceParams p);

struct ProgressBar
{
    std::atomic<int32_t> last_pos = -1;
    void update(float progress)
    {
        int barWidth = 70;

        int pos = barWidth * progress;
        if (pos != last_pos)
        {
            static std::mutex m;
            std::lock_guard<std::mutex> lock(m);
            last_pos = pos;
            std::cout << "[";
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << " %\r";
            if (pos == barWidth)
                std::cout << "\n";
            std::cout.flush();
        }
    }
};

VirtualGeometryConstructResult virtual_geometry_construct(
    torch::Tensor Positions,  
    torch::Tensor Indices,
    int MaxPartitionSize,
    std::vector<torch::Tensor> Attributes,
    torch::Device Device)
{
    NVDR_CHECK(Positions.sizes().size() == 2 && Positions.size(0) > 0 && Positions.size(1) == 3, "Positions must have shape [>0, 3]");
    NVDR_CHECK(Indices.sizes().size() == 2 && Indices.size(0) > 0 && Indices.size(1) == 3, "Indices must have shape [>0, 3]");
    nvdr_check_cpu({Positions}, __func__, "(): Inputs Positions must reside on CPU during virtual geometry construction");
    NVDR_CHECK_F32(Positions);
    NVDR_CHECK(Indices.dtype() == torch::kInt64, "(): Inputs PositionIndexes must be int64 tensors");

    VertexData Data;
    ClusterResult Result;

    Data.Indices = Indices.data_ptr<int64_t>();
    Data.NumTriangles = Indices.size(0);
    Data.NumVertices = Positions.size(0);
    Data.Positions = (vec3*)Positions.data_ptr<float>();
    
    ClusterTriangles(Data, Result, MaxPartitionSize);

    VirtualGeometryConstructResult Mesh;

    ProgressBar Bar;
    std::cout << "Extracting vertices...\n";
    auto CudaIndices = Indices.cuda();
    auto CudaPositions = Positions.cuda();
    std::vector<torch::Tensor> CudaAttributes(Attributes.size());
    for (int i = 0; i < Attributes.size(); i++)
        CudaAttributes[i] = Attributes[i].cuda();
    for (int ClusterIndex = 0; ClusterIndex < Result.Clusters.size(); ClusterIndex++)
    {
        auto& Cluster = Result.Clusters[ClusterIndex];
        VirtualGeometryCluster& OutCluster = Mesh.Clusters.emplace_back();

        uint32_t NumTriangles = Cluster.Indices.size() / 3;
        torch::TensorOptions Options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
        OutCluster.Indices = torch::from_blob(Cluster.Indices.data(), {NumTriangles, 3}, Options).clone().to(Device, true);
        Options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        OutCluster.OldTriangleIndices = torch::from_blob(Cluster.OldTriangleIndices.data(), {NumTriangles}, Options).clone().to(Device, true);

        torch::Tensor OldIndices = CudaIndices.index_select(0, OutCluster.OldTriangleIndices.cuda());    // [NumTriangles, 3]

        Options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
        torch::Tensor OldVertIndices = torch::zeros({Cluster.NumVertices}, Options).index_put({OutCluster.Indices}, OldIndices);    // [NumVertices]
        OutCluster.Positions = CudaPositions.index_select(0, OldVertIndices).to(Device, true);
        OutCluster.Attributes.resize(CudaAttributes.size());
        for (int i = 0; i < CudaAttributes.size(); i++)
            OutCluster.Attributes[i] = CudaAttributes[i].index_select(0, OldVertIndices).to(Device, true);
        OutCluster.ClusterIndex = ClusterIndex;
        Bar.update(float(ClusterIndex+1) / Result.Clusters.size());
    }
    

    return Mesh;
}

torch::Tensor virtual_geometry_frustum_cull(torch::Tensor AABBs, torch::Tensor Frustums)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(Frustums));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    NVDR_CHECK(Frustums.sizes().size()==3 && Frustums.size(1)==6 && Frustums.size(2)==4, "Frustums must have shape [>0, 6, 4]");
    NVDR_CHECK(AABBs.sizes().size()==2 && AABBs.size(0)>0 && AABBs.size(1)==6, "AABBs must have shape [>0, 6]");
    NVDR_CHECK_F32(Frustums);
    NVDR_CHECK_DEVICE(Frustums);
    NVDR_CHECK_F32(AABBs);
    NVDR_CHECK_DEVICE(AABBs);
    VirtualGeometryFrustumCullParams p;
    p.n = Frustums.size(0);
    p.numClusters = AABBs.size(0);

    p.frustums = (ViewFrustum*)Frustums.data_ptr<float>();
    p.AABBs = (BoundingBox*)AABBs.data_ptr<float>();

    torch::TensorOptions Options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
    torch::Tensor Culled = torch::zeros({p.n, p.numClusters}, Options);
    p.culled = Culled.data_ptr<bool>();


    dim3 blockSize = dim3(16, 1, 1);
    dim3 gridSize = dim3(16, std::ceil(p.numClusters / 256.0), p.n);
    void* args[] = {&p};

    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)VirtualGeometryFrustumCullKernal, gridSize, blockSize, args, 0, stream));
 
    return Culled;
}

void virtual_geometry_vertex_all_reduce(std::vector<torch::Tensor> cluster_vertices, torch::Tensor shared_verts, torch::Tensor shared_verts_offsets, int reduce_op)
{
    if (cluster_vertices.empty())
        return;
    if (shared_verts.numel() == 0)
        return;

    NVDR_CHECK_DEVICE(cluster_vertices);
    NVDR_CHECK_DEVICE(shared_verts);
    NVDR_CHECK_DEVICE(shared_verts_offsets);
    std::vector<float*> pointers;
    for (auto vertices : cluster_vertices)
    {
        NVDR_CHECK(vertices.sizes().size() == 2, "Vertices must have shape [>0, >0]");
        NVDR_CHECK(vertices.defined() && vertices.nbytes(), "Vertices must be defined and non-empty");
        pointers.push_back(vertices.data_ptr<float>());
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(cluster_vertices[0]));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    VirtualGeometryAllReduceParams p{};
    
    auto p_vertices = prepareCudaArray(pointers);
    p.vertices = (float**)p_vertices.data_ptr();
    p.matchingVerts = shared_verts.data_ptr<int>();
    p.offsetGroups = shared_verts_offsets.data_ptr<int>();
    p.numGroups = shared_verts_offsets.size(0)-1;
    p.numAttr = cluster_vertices[0].size(1);

    dim3 blockSize = dim3(64, 1, 1);
    dim3 gridSize = dim3(std::ceil(p.numGroups / 64.0), 1, 1);

    void* args[] = {&p};

    void* func_tbl[] = {
        (void*)VirtualGeometryAllReduceKernelSum, 
        (void*)VirtualGeometryAllReduceKernelAvg
    };

    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func_tbl[reduce_op], gridSize, blockSize, args, 0, stream));

}