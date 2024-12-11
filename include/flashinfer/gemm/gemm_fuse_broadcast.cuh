/*
* Copyright (c) 2024 by FlashInfer team.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
// #ifndef FLASHINFER_GEMM_GROUP_GEMM_CUH_
// #define FLASHINFER_GEMM_GROUP_GEMM_CUH_

#include <sstream>

#include "../allocator.h"
#include "../cutlass_utils.cuh"


namespace flashinfer {

namespace gemm_fuse_broadcast {









template <typename DType>
cudaError_t CutlassGemmFuseBroadcastRun(void* a, void* b, std::vector<void*> c, int rank,
                                        cudaStream_t cuda_stream) {

                                
    //status = gemm.run(stream);

    

    // if (status != cutlass::Status::kSuccess) {
    // std::ostringstream err_msg;
    // err_msg << "cutlass group_gemm.run failed: " << cutlassGetStatusString(status);
    // FLASHINFER_ERROR(err_msg.str());
    // }
    return cudaSuccess;
};


}  // namespace gemm_fuse_broadcast

}  // namespace flashinfer

// #endif  // FLASHINFER_GEMM_GROUP_GEMM_CUH_
