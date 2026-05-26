/*
 * Copyright (C) 2026 Libre Computer Project
 *
 * SPDX-License-Identifier: MIT
 *
 * cl_khr_command_buffer extension implementation for NEO.
 *
 * Status: emulation-style record/replay. Records kernel-clone snapshots
 * at clCommandNDRangeKernelKHR; clEnqueueCommandBufferKHR iterates the
 * recorded commands and submits via the existing internal NEO enqueue
 * path back-to-back without intermediate clFinish. Same perf
 * characteristics as a host-side batched-enqueue loop.
 *
 * NOT YET IMPLEMENTED: full record/replay (build the GPU command
 * stream / batch buffer at finalize time and submit via direct CSR
 * submission at replay). That delivers the additional 1.2-1.3x perf
 * gain beyond batched enqueue. Tracked as iter104+ work.
 */

#pragma once

// cl_khr_command_buffer typedefs in CL/cl_ext.h are guarded by
// CL_ENABLE_BETA_EXTENSIONS (the extension is at version 0.9.7 -- still
// "beta" per Khronos). Define before including the headers so our code
// sees the types.
#ifndef CL_ENABLE_BETA_EXTENSIONS
#define CL_ENABLE_BETA_EXTENSIONS
#endif

#include "shared/source/utilities/reference_tracked_object.h"

#include "opencl/source/api/cl_types.h"

#include "CL/cl.h"
#include "CL/cl_ext.h"

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace NEO {

class CommandQueue;
class MultiDeviceKernel;

// Forward declarations for iter104++ deep CSR submission scaffolding
// (Sprint 1A; full implementation pending per
// ~/git/claude/infra/local-llm/intel/xe-lp/audit/2026-05-02-iter104pp-csr-submission-design.md).
struct KernelOperation;
class LinearStream;
class GraphicsAllocation;
class ClCommandBuffer;
class CommandQueue;

// iter104++ Sprint 1C: per-Gen factory for populating the persistent
// batch buffer with GPGPU walker commands. Mirrors NEO's
// commandQueueFactory pattern (command_queue.cpp:69). Each per-family
// .cpp file (cl_khr_command_buffer_gen12lp.cpp etc.) registers its
// templated dispatchWalker invoker via populateFactoryTable.
using DeepCsrPopulateFn = cl_int (*)(ClCommandBuffer &cb, CommandQueue &queue);
extern DeepCsrPopulateFn deepCsrPopulateFactory[];

// State per the cl_khr_command_buffer spec (RECORDING / EXECUTABLE /
// PENDING). The header doesn't define an INVALID state at v0.9.7.
// We track an internal `invalidInternal` for diagnostic use only -- it
// surfaces externally as PENDING (closest semantic, since after a
// failed enqueue we conservatively don't allow re-execution).
enum class ClCommandBufferState {
    recording,
    executable,
    pending,
    invalidInternal,
};

// One recorded NDRange command. cl_kernel is opaque pointer to
// MultiDeviceKernel in NEO; we store the NEO-typed pointer directly
// to skip a cast on every replay.
struct RecordedNDRange {
    MultiDeviceKernel *kernelClone; // owning clone captured at record time
    cl_uint workDim;
    std::vector<size_t> globalWorkOffset;
    std::vector<size_t> globalWorkSize;
    std::vector<size_t> localWorkSize;
    bool hasLocalSize;
    bool hasOffset;
};

class ClCommandBuffer {
  public:
    ClCommandBuffer(CommandQueue *queue, const cl_command_buffer_properties_khr *properties);
    ~ClCommandBuffer();

    // Refcounting (extension objects are retained/released by CL spec).
    void retain() { refcount.fetch_add(1, std::memory_order_relaxed); }
    bool release() {
        return refcount.fetch_sub(1, std::memory_order_acq_rel) == 1;
    }

    ClCommandBufferState getState() const { return state; }
    CommandQueue *getQueue() const { return associatedQueue; }

    // Record an NDRange kernel command. Clones the kernel internally so
    // post-record kernel-arg changes by the user do NOT affect this
    // command buffer. If `mutableHandleOut` is non-null, populates it
    // with a stable pointer to the recorded entry (for later use with
    // updateMutableCommands). Pointer remains valid for cmdbuf lifetime
    // since recordedCommands is std::deque.
    cl_int recordNDRangeKernel(CommandQueue *queue,
                               cl_kernel kernel,
                               cl_uint workDim,
                               const size_t *globalWorkOffset,
                               const size_t *globalWorkSize,
                               const size_t *localWorkSize,
                               cl_mutable_command_khr *mutableHandleOut = nullptr);

    // Transition recording -> executable. After finalize, no more
    // commands can be recorded.
    cl_int finalize();

    // Replay: iterate recorded commands and enqueue back-to-back on
    // `queue` (or the associated queue if NULL). Returns CL_INVALID_*
    // if the command buffer is not executable.
    cl_int enqueue(CommandQueue *queue,
                   cl_uint numEventsInWaitList,
                   const cl_event *eventWaitList,
                   cl_event *event);

    // cl_khr_command_buffer_mutable_dispatch: update kernel args, work
    // sizes, and global offsets on previously-recorded commands without
    // re-recording the cmdbuf. Each config of type
    // CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR carries a
    // cl_mutable_command_khr handle (opaque pointer to a stored
    // RecordedNDRange) plus a list of per-arg updates.
    //
    // Handle stability: recordedCommands is std::deque, which preserves
    // pointer/iterator validity across back insertions. Mutable handles
    // (pointers into the deque) remain valid for the cmdbuf lifetime.
    cl_int updateMutableCommands(cl_uint numConfigs,
                                  const cl_command_buffer_update_type_khr *configTypes,
                                  const void **configs);

    // iter104++ deep CSR submission scaffolding (Sprint 1A-1C).
    // Persistent batch buffer state: at finalize, deepCsrInit() routes
    // through deepCsrPopulateFactory[CoreFamily] to walk recordedCommands
    // and emit GPGPU walker commands into the owned LinearStream via
    // HardwareInterface<Family>::dispatchWalker (with blockedCommandsData
    // hijack). At enqueue, deepCsrSubmit() submits the persistent
    // BatchBuffer once via CommandStreamReceiver::submitBatchBuffer.
    // Gated on env var CL_CMDBUF_DEEP_CSR=1; default OFF until
    // mutable-update integration (Sprint 2) lands.
    cl_int deepCsrInit(CommandQueue *queue);
    cl_int deepCsrSubmit(CommandQueue *queue,
                          cl_uint numEventsInWaitList,
                          const cl_event *eventWaitList,
                          cl_event *event);

    // Accessors for the per-Gen populate function (Sprint 1D will
    // implement the populate body in cl_khr_command_buffer_<gen>.cpp).
    const std::deque<RecordedNDRange> &getRecordedCommands() const { return recordedCommands; }
    KernelOperation *getDeepCsrKernelOp() { return deepCsrKernelOp.get(); }
    void setDeepCsrInitialized(bool v) { deepCsrInitialized = v; }

  private:
    std::atomic<uint32_t> refcount{1};
    CommandQueue *associatedQueue;
    ClCommandBufferState state{ClCommandBufferState::recording};
    std::vector<cl_command_buffer_properties_khr> propertiesStorage;
    // std::deque (NOT std::vector) because cl_mutable_command_khr handles
    // are pointers INTO this container; deque guarantees iterator/pointer
    // stability across back insertions, vector does not.
    std::deque<RecordedNDRange> recordedCommands;
    std::mutex stateMutex;

    // iter104++ deep CSR submission state (Sprint 1A scaffolding).
    // deepCsrKernelOp owns the persistent LinearStream; deepCsrInitialized
    // tracks whether the GPGPU command stream has been emitted yet.
    std::unique_ptr<KernelOperation> deepCsrKernelOp;
    bool deepCsrInitialized{false};
};

} // namespace NEO

// API entry points (CL spec calling convention).
extern "C" {

CL_API_ENTRY cl_command_buffer_khr CL_API_CALL
clCreateCommandBufferKHR(cl_uint numQueues,
                         const cl_command_queue *queues,
                         const cl_command_buffer_properties_khr *properties,
                         cl_int *errcodeRet);

CL_API_ENTRY cl_int CL_API_CALL
clFinalizeCommandBufferKHR(cl_command_buffer_khr commandBuffer);

CL_API_ENTRY cl_int CL_API_CALL
clRetainCommandBufferKHR(cl_command_buffer_khr commandBuffer);

CL_API_ENTRY cl_int CL_API_CALL
clReleaseCommandBufferKHR(cl_command_buffer_khr commandBuffer);

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCommandBufferKHR(cl_uint numQueues,
                          cl_command_queue *queues,
                          cl_command_buffer_khr commandBuffer,
                          cl_uint numEventsInWaitList,
                          const cl_event *eventWaitList,
                          cl_event *event);

CL_API_ENTRY cl_int CL_API_CALL
clCommandNDRangeKernelKHR(cl_command_buffer_khr commandBuffer,
                          cl_command_queue commandQueue,
                          const cl_command_properties_khr *properties,
                          cl_kernel kernel,
                          cl_uint workDim,
                          const size_t *globalWorkOffset,
                          const size_t *globalWorkSize,
                          const size_t *localWorkSize,
                          cl_uint numSyncPointsInWaitList,
                          const cl_sync_point_khr *syncPointWaitList,
                          cl_sync_point_khr *syncPoint,
                          cl_mutable_command_khr *mutableHandle);

CL_API_ENTRY cl_int CL_API_CALL
clGetCommandBufferInfoKHR(cl_command_buffer_khr commandBuffer,
                          cl_command_buffer_info_khr paramName,
                          size_t paramValueSize,
                          void *paramValue,
                          size_t *paramValueSizeRet);

// cl_khr_command_buffer_mutable_dispatch sub-extension entry point.
// Updates kernel args, work sizes, and global offsets on previously-
// recorded commands. Each config is a cl_mutable_dispatch_config_khr*
// (when configTypes[i] == CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR
//  == 0). Returns CL_INVALID_MUTABLE_COMMAND_KHR if a cfg.command
// handle does not refer to a recorded command in this cmdbuf.
CL_API_ENTRY cl_int CL_API_CALL
clUpdateMutableCommandsKHR(cl_command_buffer_khr commandBuffer,
                            cl_uint numConfigs,
                            const cl_command_buffer_update_type_khr *configTypes,
                            const void **configs);

} // extern "C"

// Map between cl_command_buffer_khr opaque pointer and NEO::ClCommandBuffer.
// We use a simple identity cast: the opaque type IS the ClCommandBuffer
// pointer (no separate dispatch table since the CL spec doesn't require
// per-instance dispatch for extension objects).
namespace NEO {
inline ClCommandBuffer *toNeoCommandBuffer(cl_command_buffer_khr handle) {
    return reinterpret_cast<ClCommandBuffer *>(handle);
}
inline cl_command_buffer_khr toClHandle(ClCommandBuffer *cb) {
    return reinterpret_cast<cl_command_buffer_khr>(cb);
}
} // namespace NEO
