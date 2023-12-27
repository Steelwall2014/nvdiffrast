#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../common/framework.h"
#include "../common/virtual_geometry_partition.h"
#include "../common/virtual_geometry.h"
#include "torch_types.h"

void VirtualGeometryFrustumCullKernal(VirtualGeometryFrustumCullParams p);
void VirtualGeometryAccumulateGradKernel(VirtualGeometryAccumulateGradParams p);

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
    std::vector<torch::Tensor> Attributes)
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
        OutCluster.Indices = torch::from_blob(Cluster.Indices.data(), {NumTriangles, 3}, Options).cuda();
        Options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        OutCluster.OldTriangleIndices = torch::from_blob(Cluster.OldTriangleIndices.data(), {NumTriangles}, Options).cuda();

        torch::Tensor OldIndices = CudaIndices.index_select(0, OutCluster.OldTriangleIndices);    // [NumTriangles, 3]

        Options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
        torch::Tensor OldVertIndices = torch::zeros({Cluster.NumVertices}, Options).index_put({OutCluster.Indices}, OldIndices);    // [NumVertices]
        OutCluster.Positions = CudaPositions.index_select(0, OldVertIndices);
        OutCluster.Attributes.resize(CudaAttributes.size());
        for (int i = 0; i < CudaAttributes.size(); i++)
            OutCluster.Attributes[i] = CudaAttributes[i].index_select(0, OldVertIndices);
        OutCluster.ClusterIndex = ClusterIndex;
        Bar.update(float(ClusterIndex+1) / Result.Clusters.size());
    }
    
    Mesh.MatchingVertices = Result.MatchingVertices;

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

void virtual_geometry_accumulate_grad(std::vector<torch::Tensor> Clusters, std::vector<std::vector<std::tuple<int, int>>> MatchingVertices)
{
    std::vector<float*> cuda_grads;
    std::vector<float*> cpu_grads;
    std::optional<torch::Tensor> cuda_tensor = std::nullopt;
    std::optional<torch::Tensor> cpu_tensor = std::nullopt;

    for (auto& cluster : Clusters)
    {
        NVDR_CHECK(cluster.sizes().size() == 2 && cluster.size(0) > 0 && cluster.size(1) > 0, "The attribute of each cluster must have shape [>0, >0]");
        torch::Tensor grad = cluster.grad();
        bool has_grad = grad.defined() && grad.nbytes();
        if (!has_grad)
        {
            cpu_grads.push_back(NULL);
        }
        else 
        {
            if (at::cuda::check_device({cluster}))
            {
                cuda_tensor = cluster;
                cuda_grads.push_back(grad.data_ptr<float>());
            }
            else 
            {
                cpu_tensor = cluster;
                cpu_grads.push_back(grad.data_ptr<float>());
            }
        }
    }

    if (cuda_tensor)
    {
        const at::cuda::OptionalCUDAGuard device_guard(device_of(cuda_tensor.value()));

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        VirtualGeometryAccumulateGradParams p{};
        
        auto p_grad = prepareCudaArray(cuda_grads);
        p.grad = (float**)p_grad.data_ptr();
        std::vector<int32_t> matchingVerts;
        std::vector<int32_t> offsetGroups;
        for (auto& Group : MatchingVertices)
        {
            offsetGroups.push_back(matchingVerts.size());
            for (auto& Vertex : Group)
            {
                matchingVerts.push_back(std::get<0>(Vertex));   // cluster index
                matchingVerts.push_back(std::get<1>(Vertex));   // vertex index
            }
        }
        offsetGroups.push_back(matchingVerts.size());
        auto p_matchingVerts = prepareCudaArray(matchingVerts);
        auto p_offsetGroups = prepareCudaArray(offsetGroups);
        p.matchingVerts = (int*)p_matchingVerts.data_ptr();
        p.offsetGroups = (int*)p_offsetGroups.data_ptr();
        p.numGroups = MatchingVertices.size();
        p.numAttr = cuda_tensor->size(1);

        dim3 blockSize = dim3(32, 1, 1);
        dim3 gridSize = dim3(16, std::ceil(p.numGroups / 512.0), 1);

        void* args[] = {&p};
        NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)VirtualGeometryAccumulateGradKernel, gridSize, blockSize, args, 0, stream));

    }

    if (cpu_tensor)
    {
        for (auto& Group : MatchingVertices)
        {
            int numVerts = Group.size();
            int numAttr = cpu_tensor->size(1);
            for (int j = 0; j < numVerts; j++)
            {
                float grad = 0.f;
                for (int i = 0; i < numVerts; i++)
                {
                    int clusterIndex = std::get<0>(Group[i]);
                    int vertexIndex = std::get<1>(Group[i]);
                    if (cpu_grads[clusterIndex])
                        grad += cpu_grads[clusterIndex][vertexIndex * numAttr + j];
                }
                for (int i = 0; i < numVerts; i++)
                {
                    int clusterIndex = std::get<0>(Group[i]);
                    int vertexIndex = std::get<1>(Group[i]);
                    if (cpu_grads[clusterIndex])
                        cpu_grads[clusterIndex][vertexIndex * numAttr + j] = grad;
                }
            }
        }

    }

}