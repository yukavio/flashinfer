#include <flashinfer/gemm/gemm_fuse_broadcast.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;
using namespace flashinfer::gemm_fuse_broadcast;

void CutlassGemmFuseBroadcast(at::Tensor A, at::Tensor B, std::vector<at::Tensor> C, int rank,
    int64_t cuda_stream) {

cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
DISPATCH_PYTORCH_DTYPE_TO_CTYPE(A.scalar_type(), c_type, [&] {
    using cutlass_t = typename cutlass_dtype<c_type>::value;
    std::vector<void *> c_ptr;
    for(const auto& t : C){
        c_ptr.push_back(t.data_ptr());
    }
    auto status = CutlassGemmFuseBroadcastRun<cutlass_t>(
        A.data_ptr(), B.data_ptr(), c_ptr, rank, stream);
    TORCH_CHECK(status == cudaSuccess,
                "Failed to run CutlassGemmFuseBroadcast: ", cudaGetErrorString(status));
    return true;
});
}
 