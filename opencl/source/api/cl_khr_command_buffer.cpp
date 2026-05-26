/*
 * Copyright (C) 2026 Libre Computer Project
 *
 * SPDX-License-Identifier: MIT
 *
 * cl_khr_command_buffer extension implementation for NEO.
 *
 * Architecture: emulation-style (record + replay via per-command
 * clEnqueueNDRangeKernel). See header for status notes.
 */

// CL_ENABLE_BETA_EXTENSIONS must be defined BEFORE the header includes
// since the cl_khr_command_buffer typedefs in CL/cl_ext.h are gated on it.
#ifndef CL_ENABLE_BETA_EXTENSIONS
#define CL_ENABLE_BETA_EXTENSIONS
#endif

#include "opencl/source/api/cl_khr_command_buffer.h"

#include "opencl/source/cl_device/cl_device.h"
#include "opencl/source/command_queue/command_queue.h"
#include "opencl/source/helpers/task_information.h"
#include "shared/source/command_container/cmdcontainer.h"
#include "shared/source/command_stream/command_stream_receiver.h"
#include "shared/source/command_stream/linear_stream.h"
#include "shared/source/command_stream/submissions_aggregator.h"
#include "shared/source/device/device.h"
#include "shared/source/helpers/constants.h"
#include "shared/source/helpers/engine_node_helper.h"
#include "shared/source/indirect_heap/indirect_heap.h"
#include "shared/source/memory_manager/graphics_allocation.h"
#include "shared/source/memory_manager/internal_allocation_storage.h"
#include "shared/source/memory_manager/residency_container.h"
#include "shared/source/program/kernel_info.h"

#include "opencl/source/kernel/kernel.h"
#include "opencl/source/kernel/multi_device_kernel.h"
#include "opencl/source/context/context.h"
#include "opencl/source/helpers/cl_validators.h"
#include "opencl/source/kernel/kernel.h"
#include "opencl/source/kernel/multi_device_kernel.h"
#include "opencl/source/platform/platform.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace NEO {

// iter104++ Sprint 1C: per-Gen factory table for the deep-CSR
// populate function. Each per-family .cpp file (Gen12LP, Xe-HPG, Xe2,
// Xe3) registers its templated populate function via populateFactoryTable.
// Default-zero entries cause deepCsrInit to leave deepCsrInitialized
// false -- the env-var gate in enqueue() then falls through to the
// existing batchedDispatch path.
DeepCsrPopulateFn deepCsrPopulateFactory[NEO::maxCoreEnumValue] = {};

ClCommandBuffer::ClCommandBuffer(CommandQueue *queue,
                              const cl_command_buffer_properties_khr *properties)
    : associatedQueue(queue) {
    if (properties != nullptr) {
        // Per spec, properties is a 0-terminated list of (name, value) pairs.
        for (auto p = properties; *p != 0; ) {
            propertiesStorage.push_back(*p++); // name
            propertiesStorage.push_back(*p++); // value
        }
        propertiesStorage.push_back(0); // terminator
    }
    if (associatedQueue != nullptr) {
        // Retain the associated queue for the lifetime of the buffer.
        // CommandQueue derives from BaseObject<_cl_command_queue>; cast to
        // the opaque cl_command_queue handle for the public retain API.
        clRetainCommandQueue(static_cast<cl_command_queue>(associatedQueue));
    }
}

ClCommandBuffer::~ClCommandBuffer() {
    // Release recorded kernel clones.
    for (auto &rec : recordedCommands) {
        if (rec.kernelClone != nullptr) {
            clReleaseKernel(static_cast<cl_kernel>(rec.kernelClone));
        }
    }
    if (associatedQueue != nullptr) {
        clReleaseCommandQueue(static_cast<cl_command_queue>(associatedQueue));
    }
}

cl_int ClCommandBuffer::recordNDRangeKernel(CommandQueue *queue,
                                           cl_kernel kernel,
                                           cl_uint workDim,
                                           const size_t *globalWorkOffset,
                                           const size_t *globalWorkSize,
                                           const size_t *localWorkSize,
                                           cl_mutable_command_khr *mutableHandleOut) {
    std::lock_guard<std::mutex> lock(stateMutex);
    if (state != ClCommandBufferState::recording) {
        return CL_INVALID_OPERATION;
    }
    if (kernel == nullptr) {
        return CL_INVALID_KERNEL;
    }
    if (workDim < 1 || workDim > 3 || globalWorkSize == nullptr) {
        return CL_INVALID_VALUE;
    }

    // Clone the kernel to capture the current arg state. The clone owns
    // an independent copy of the kernelArguments table; subsequent
    // clSetKernelArg calls on the original do NOT affect the clone.
    cl_int err = CL_SUCCESS;
    cl_kernel clone = clCloneKernel(kernel, &err);
    if (err != CL_SUCCESS) {
        return err;
    }

    RecordedNDRange rec;
    rec.kernelClone = castToObject<MultiDeviceKernel>(clone);
    rec.workDim = workDim;
    rec.globalWorkSize.assign(globalWorkSize, globalWorkSize + workDim);
    rec.hasLocalSize = (localWorkSize != nullptr);
    if (rec.hasLocalSize) {
        rec.localWorkSize.assign(localWorkSize, localWorkSize + workDim);
    }
    rec.hasOffset = (globalWorkOffset != nullptr);
    if (rec.hasOffset) {
        rec.globalWorkOffset.assign(globalWorkOffset, globalWorkOffset + workDim);
    }
    recordedCommands.push_back(std::move(rec));
    // The mutable handle is the address of the entry in our deque.
    // deque guarantees this pointer remains valid across subsequent
    // back-insertions for the cmdbuf lifetime (header invariant).
    if (mutableHandleOut != nullptr) {
        *mutableHandleOut = reinterpret_cast<cl_mutable_command_khr>(&recordedCommands.back());
    }
    return CL_SUCCESS;
}

cl_int ClCommandBuffer::updateMutableCommands(cl_uint numConfigs,
                                               const cl_command_buffer_update_type_khr *configTypes,
                                               const void **configs) {
    if (numConfigs == 0) return CL_SUCCESS;
    if (configTypes == nullptr || configs == nullptr) return CL_INVALID_VALUE;

    std::lock_guard<std::mutex> lock(stateMutex);
    // Spec allows update on EXECUTABLE; we also allow PENDING (stays
    // queued; updates apply on next enqueue).
    if (state != ClCommandBufferState::executable &&
        state != ClCommandBufferState::pending) {
        return CL_INVALID_OPERATION;
    }

    for (cl_uint i = 0; i < numConfigs; i++) {
        // Per Khronos: type 0 == CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR.
        if (configTypes[i] != 0) {
            return CL_INVALID_VALUE;
        }
        auto cfg = static_cast<const cl_mutable_dispatch_config_khr *>(configs[i]);
        if (cfg == nullptr || cfg->command == nullptr) {
            return CL_INVALID_MUTABLE_COMMAND_KHR;
        }

        // Validate that cfg->command actually points to one of OUR
        // recorded entries. The deque-pointer-stability invariant means
        // any entry's address is still valid; we just verify by linear
        // scan that the pointer is in our owned set. O(N) per cfg but
        // numConfigs is typically 1-3 in practice (per-token rebinds).
        RecordedNDRange *rec = nullptr;
        for (auto &r : recordedCommands) {
            if (&r == reinterpret_cast<const RecordedNDRange *>(cfg->command)) {
                rec = &r;
                break;
            }
        }
        if (rec == nullptr) {
            return CL_INVALID_MUTABLE_COMMAND_KHR;
        }

        // Per-arg updates: clSetKernelArg writes the kernel-object's
        // persistent slot state. The clone owned by RecordedNDRange is
        // re-used at every replay, so updates persist until the next
        // updateMutableCommands or destruction.
        for (cl_uint a = 0; a < cfg->num_args; a++) {
            const auto &arg = cfg->arg_list[a];
            cl_int err = clSetKernelArg(static_cast<cl_kernel>(rec->kernelClone),
                                        arg.arg_index, arg.arg_size, arg.arg_value);
            if (err != CL_SUCCESS) return err;
        }

        // Work-size / offset updates: target rec->workDim slots in our
        // stored vectors. Spec requires all three (gwo/gws/lws) to use
        // the same workDim as recorded; updating workDim itself is not
        // supported in this minimal impl.
        if (cfg->global_work_offset != nullptr) {
            rec->globalWorkOffset.assign(cfg->global_work_offset,
                                         cfg->global_work_offset + rec->workDim);
            rec->hasOffset = true;
        }
        if (cfg->global_work_size != nullptr) {
            rec->globalWorkSize.assign(cfg->global_work_size,
                                       cfg->global_work_size + rec->workDim);
            static const char *dbg = std::getenv("CL_CMDBUF_DEBUG");
            if (dbg && dbg[0] == '1') {
                fprintf(stderr, "[NEO-CMDBUF] update rec=%p workDim=%u gws=[%zu,%zu,%zu]\n",
                        (void*)rec, rec->workDim,
                        rec->workDim>=1?rec->globalWorkSize[0]:0,
                        rec->workDim>=2?rec->globalWorkSize[1]:0,
                        rec->workDim>=3?rec->globalWorkSize[2]:0);
            }
        }
        if (cfg->local_work_size != nullptr) {
            rec->localWorkSize.assign(cfg->local_work_size,
                                      cfg->local_work_size + rec->workDim);
            rec->hasLocalSize = true;
        }

        // SVM args + exec_info_list: not currently used by gaema's mega
        // path; reject if requested rather than silently skip.
        if (cfg->num_svm_args != 0 || cfg->num_exec_infos != 0) {
            return CL_INVALID_VALUE;
        }
    }
    return CL_SUCCESS;
}

cl_int ClCommandBuffer::finalize() {
    std::lock_guard<std::mutex> lock(stateMutex);
    if (state != ClCommandBufferState::recording) {
        return CL_INVALID_OPERATION;
    }
    state = ClCommandBufferState::executable;

    // iter104++ Sprint 1B (PARTIAL): when CL_CMDBUF_DEEP_CSR=1, allocate
    // the persistent KernelOperation+LinearStream now so subsequent
    // enqueues go through deepCsrSubmit. Stays default-OFF since
    // Sprint 1C (per-Gen factory dispatch for dispatchWalker) is not
    // landed -- without it, deepCsrInit currently allocates the buffer
    // but doesn't populate; deepCsrInitialized stays false; enqueue's
    // env-var gate sees the unpopulated state and falls through to
    // batchedDispatch.
    const char *deepCsrEnv = std::getenv("CL_CMDBUF_DEEP_CSR");
    if (deepCsrEnv && deepCsrEnv[0] == '1' && associatedQueue != nullptr) {
        deepCsrInit(associatedQueue);
        // Return CL_SUCCESS even if deepCsrInit returns non-success --
        // we want the cmdbuf to remain functional via the fallback path.
    }

    return CL_SUCCESS;
}

cl_int ClCommandBuffer::enqueue(CommandQueue *queue,
                              cl_uint numEventsInWaitList,
                              const cl_event *eventWaitList,
                              cl_event *event) {
    {
        std::lock_guard<std::mutex> lock(stateMutex);
        if (state != ClCommandBufferState::executable &&
            state != ClCommandBufferState::pending) {
            return CL_INVALID_OPERATION;
        }
        state = ClCommandBufferState::pending;
    }

    CommandQueue *targetQueue = (queue != nullptr) ? queue : associatedQueue;
    if (targetQueue == nullptr) {
        std::lock_guard<std::mutex> lock(stateMutex);
        state = ClCommandBufferState::executable;
        return CL_INVALID_COMMAND_QUEUE;
    }

    cl_int err = CL_SUCCESS;
    const size_t numCommands = recordedCommands.size();

    // iter104++ Sprint 1A: deep CSR submission path, gated on env var
    // CL_CMDBUF_DEEP_CSR=1. When enabled and infrastructure is ready,
    // route to deepCsrSubmit which uses the persistent KernelOperation
    // populated by deepCsrInit (called from finalize). Default OFF
    // because mutable-update integration (Sprint 2) is not yet landed
    // -- without it the path can't service clUpdateMutableCommandsKHR
    // calls. The batchedDispatch path below is the production default.
    const char *deepCsrEnv = std::getenv("CL_CMDBUF_DEEP_CSR");
    if (deepCsrEnv && deepCsrEnv[0] == '1' && deepCsrInitialized) {
        cl_int submitResult = deepCsrSubmit(targetQueue, numEventsInWaitList,
                                              eventWaitList, event);
        // ONLY return submit's result on actual GPU submission (success
        // OR a non-recoverable error). If submit returns
        // CL_INVALID_OPERATION it means the inner FORCE gate or
        // residency setup short-circuited -- fall through to
        // batchedDispatch below so production stays working while we
        // iterate Sprints 1G+.
        if (submitResult != CL_INVALID_OPERATION) {
            return submitResult;
        }
    }
    // deepCsrInitialized stays false until Sprint 1C lands the per-Gen
    // dispatchWalker factory; until then, even with CL_CMDBUF_DEEP_CSR=1
    // we fall through to the working batchedDispatch path below.

    // Switch CSR to batchedDispatch mode so the N enqueueKernel calls
    // accumulate in the CSR's batch list instead of submitting per-call.
    // Then flushBatchedSubmissions() at the end issues ONE GPU
    // submission for all 90 launches. This is the minimal-viable
    // single-shot-submit path without rewriting the dispatchWalker
    // pipeline (full deep CSR submission per task #46 audit
    // 2026-05-02-iter104pp-csr-submission-design.md).
    auto &csr = targetQueue->getGpgpuCommandStreamReceiver();
    auto savedDispatchMode = csr.getDispatchMode();
    csr.overrideDispatchPolicy(DispatchMode::batchedDispatch);

    // iter104++ Option B (Sprint 2X): hoist per-API-layer call-invariant
    // ownership + CSR-client setup OUT of the per-kernel loop. Each
    // inner enqueueHandler (via CommandQueue::enqueueKernel ->
    // CommandQueueHw<>::enqueueHandler) re-acquires queue ownership,
    // CSR ownership, and re-runs registerGpgpuCsrClient -- but for an
    // N-kernel cmdbuf replay these are call-invariant: the same queue,
    // the same CSR, the same client registration apply across all N.
    //
    // Queue ownership uses BaseObject's recursive ownage counter
    // (base_object.h::takeOwnership: same-thread re-entry just bumps
    // recursiveOwnageCounter). CSR ownership is std::recursive_mutex
    // (command_stream_receiver.h: using MutexType = std::recursive_mutex).
    // Pre-acquiring both at the cmdbuf layer converts the per-call
    // mutex acquire+release into a cheap counter bump on the inner
    // call, while preserving all ordering semantics for any other
    // threads contending on the same queue/CSR.
    //
    // registerGpgpuCsrClient is already idempotent (its first invocation
    // sets gpgpuCsrClientRegistered; subsequent calls short-circuit), so
    // its per-iteration cost is one well-predicted branch. It's also
    // protected on CommandQueue, so hoisting would require either a
    // friend declaration or visibility change -- both outside Option B's
    // "no NEO surgery" scope. Leave it inside enqueueHandler; the
    // ownership-lock hoist below is where the measurable per-call saving
    // lives (recursive mutex acquire+release replaced by counter bump).
    //
    // This is the DIAGNOSTIC measurement step: if Gemma 4 E2B
    // FORCE_SYMBOLIC=1 moves >=1.5 t/s with this hoist, per-API-layer
    // overhead is the wall. If it stays <=1.2 t/s, the wall lives
    // inside per-flushTask state cost (SBA / pipeline select / preamble
    // / VFE / preemption) and a deeper re-architecture is required.
    TakeOwnershipWrapper<CommandQueue> queueOwnership(*targetQueue);
    auto csrOwnership = csr.obtainUniqueOwnership();

    // Bypass the clEnqueueNDRangeKernel public API: each public-API call
    // pays TRACING_ENTER/EXIT, API_ENTER, castToObject lookups, ownership
    // wrapper, and gtpin notification (~5-10us per call). On a 90-kernel
    // replay that's 450-900us of pure host overhead. Resolve the kernel
    // once per record (multiDeviceKernel -> Kernel for queue's device)
    // and call CommandQueue::enqueueKernel directly.
    const auto rootDeviceIndex = targetQueue->getDevice().getRootDeviceIndex();
    for (size_t i = 0; i < numCommands; i++) {
        const auto &rec = recordedCommands[i];
        const size_t *gwo = rec.hasOffset ? rec.globalWorkOffset.data() : nullptr;
        const size_t *lws = rec.hasLocalSize ? rec.localWorkSize.data() : nullptr;

        cl_uint thisNumEvents = (i == 0) ? numEventsInWaitList : 0;
        const cl_event *thisWaitList = (i == 0) ? eventWaitList : nullptr;
        cl_event *thisEvent = (i == numCommands - 1) ? event : nullptr;

        Kernel *pKernel = rec.kernelClone->getKernel(rootDeviceIndex);
        {
            static const char *dbg = std::getenv("CL_CMDBUF_DEBUG");
            if (dbg && dbg[0] == '1') {
                fprintf(stderr, "[NEO-CMDBUF] enqueue i=%zu rec=%p workDim=%u gws=[%zu,%zu,%zu] lws=%s\n",
                        i, (const void*)&rec, rec.workDim,
                        rec.workDim>=1?rec.globalWorkSize[0]:0,
                        rec.workDim>=2?rec.globalWorkSize[1]:0,
                        rec.workDim>=3?rec.globalWorkSize[2]:0,
                        lws ? "yes" : "no");
            }
        }
        err = targetQueue->enqueueKernel(pKernel, rec.workDim, gwo,
                                          rec.globalWorkSize.data(), lws,
                                          thisNumEvents, thisWaitList,
                                          thisEvent);

        if (err != CL_SUCCESS) {
            csr.overrideDispatchPolicy(savedDispatchMode);
            // queueOwnership + csrOwnership released by RAII on return.
            std::lock_guard<std::mutex> lock(stateMutex);
            state = ClCommandBufferState::invalidInternal;
            return err;
        }
    }

    // Flush the accumulated batch in ONE submission.
    csr.flushBatchedSubmissions();
    csr.overrideDispatchPolicy(savedDispatchMode);
    // queueOwnership + csrOwnership released by RAII at scope exit.

    {
        std::lock_guard<std::mutex> lock(stateMutex);
        state = ClCommandBufferState::executable;
    }
    return CL_SUCCESS;
}

cl_int ClCommandBuffer::deepCsrInit(CommandQueue *queue) {
    // Sprint 1B (PARTIAL): allocate the persistent KernelOperation +
    // LinearStream. The dispatchWalker invocation per recorded command
    // requires per-Gen template instantiation routed via a NEO factory
    // pattern (similar to commandQueueFactory[CoreFamily]) -- TODO
    // Sprint 1C below. Without that the command stream stays empty and
    // submission would be a no-op; deepCsrInitialized stays false so
    // the env-var-gated path in enqueue() correctly falls through to
    // the working batchedDispatch implementation.
    if (queue == nullptr) {
        return CL_INVALID_COMMAND_QUEUE;
    }
    if (deepCsrKernelOp != nullptr) {
        // Already initialized. Idempotent.
        return CL_SUCCESS;
    }

    // Match NEO's canonical KernelOperation construction pattern from
    // command_queue_hw.h:523-531 (used by NEO's blocked-enqueue path).
    constexpr size_t additionalAllocationSize = CSRequirements::csOverfetchSize;
    constexpr size_t allocationSize = MemoryConstants::pageSize64k - CSRequirements::csOverfetchSize;

    auto *commandStream = new LinearStream();
    auto &gpgpuCsr = queue->getGpgpuCommandStreamReceiver();
    gpgpuCsr.ensureCommandBufferAllocation(*commandStream, allocationSize,
                                            additionalAllocationSize);

    deepCsrKernelOp = std::make_unique<KernelOperation>(
        commandStream, *gpgpuCsr.getInternalAllocationStorage());

    // Sprint 1C: dispatch through per-Gen factory to populate the
    // command stream with GPGPU walker commands. If no per-Gen
    // populate function is registered (e.g., not yet implemented for
    // this silicon), leave deepCsrInitialized false so enqueue's
    // env-var gate falls through to batchedDispatch.
    auto coreFamily = queue->getDevice().getRenderCoreFamily();
    if (coreFamily < NEO::maxCoreEnumValue && deepCsrPopulateFactory[coreFamily] != nullptr) {
        cl_int populateResult = deepCsrPopulateFactory[coreFamily](*this, *queue);
        if (populateResult == CL_SUCCESS) {
            deepCsrInitialized = true;
        }
        // else: leave deepCsrInitialized=false; enqueue falls through
    }

    // TODO Sprint 1D: per-Gen populate function bodies (one per family).
    // NEO has commandQueueFactory[CoreFamily] (command_queue.cpp:69) as
    // the canonical pattern -- mirror it for our deepCsr path:
    //   1. Add a global table:
    //        using DeepCsrPopulateFn = cl_int(*)(ClCommandBuffer&, CommandQueue&);
    //        DeepCsrPopulateFn deepCsrPopulateFactory[NEO::maxCoreEnumValue] = {};
    //   2. Per-family .cpp instantiator (one per family:
    //        cl_khr_command_buffer_xe_lp.cpp, _xe_hpg.cpp, _xe2_hpg.cpp,
    //        _xe3_hpc.cpp): explicitly instantiates dispatchWalker<Family>
    //        and registers via static initializer.
    //   3. deepCsrInit dispatch:
    //        auto fn = deepCsrPopulateFactory[queue->getDevice().getRenderCoreFamily()];
    //        if (!fn) return CL_INVALID_OPERATION;
    //        return fn(*this, *queue);
    //   4. The factory function walks recordedCommands, builds
    //      MultiDispatchInfo per command (per dispatch_walker_tests.cpp:165-179
    //      pattern), invokes HardwareInterface<Family>::dispatchWalker<DefaultWalkerType>
    //      with walkerArgs.blockedCommandsData = deepCsrKernelOp.get().
    //   5. Sets deepCsrInitialized = true on success.
    //
    // Estimated Sprint 1C effort: 1-2 days (the per-Gen .cpp files are
    // mostly boilerplate around the templated call). The dispatchWalker
    // call itself is well-documented by dispatch_walker_tests.cpp -- the
    // per-Gen wiring is just registry plumbing.

    return CL_SUCCESS; // KernelOperation+LinearStream allocated; Sprint 1C will populate
}

cl_int ClCommandBuffer::deepCsrSubmit(CommandQueue *queue,
                                       cl_uint numEventsInWaitList,
                                       const cl_event *eventWaitList,
                                       cl_event *event) {
    // Sprint 1F: build BatchBuffer wrapping the persistent stream and
    // submit via CommandStreamReceiver::submitBatchBuffer. ONE GPU
    // submission per replay -- the single biggest projected win of
    // iter104++ deep CSR (replaces N internal enqueueKernel calls
    // with 1 batched submission).
    //
    // Defensive double-gate: requires CL_CMDBUF_DEEP_CSR=1 (already
    // checked by caller in ClCommandBuffer::enqueue) PLUS
    // CL_CMDBUF_DEEP_CSR_FORCE=1 to actually invoke submitBatchBuffer.
    // Without the FORCE gate, returns CL_INVALID_OPERATION and the
    // caller falls through to batchedDispatch. This lets us land the
    // submission code path without breaking production -- enable FORCE
    // for testing, leave it off for safety until residency tracking +
    // event handling are validated.
    (void)numEventsInWaitList;
    (void)eventWaitList;
    (void)event;

    if (queue == nullptr || deepCsrKernelOp == nullptr ||
        deepCsrKernelOp->commandStream == nullptr) {
        return CL_INVALID_OPERATION;
    }

    const char *forceEnv = std::getenv("CL_CMDBUF_DEEP_CSR_FORCE");
    if (!(forceEnv && forceEnv[0] == '1')) {
        // Caller will fall through to batchedDispatch.
        return CL_INVALID_OPERATION;
    }

    auto *stream = deepCsrKernelOp->commandStream.get();
    auto *commandBufferAllocation = stream->getGraphicsAllocation();
    if (commandBufferAllocation == nullptr) {
        return CL_INVALID_OPERATION;
    }

    BatchBuffer batchBuffer{};
    batchBuffer.commandBufferAllocation = commandBufferAllocation;
    batchBuffer.startOffset = 0;
    batchBuffer.usedSize = stream->getUsed();
    batchBuffer.stream = stream;
    batchBuffer.taskStartAddress = commandBufferAllocation->getGpuAddress();
    batchBuffer.lowPriority = false;
    batchBuffer.throttle = QueueThrottle::MEDIUM;
    batchBuffer.sliceCount = QueueSliceCount::defaultSliceCount;
    batchBuffer.hasStallingCmds = true; // we appended a PIPE_CONTROL barrier

    // Sprint 1H+1: build a complete ResidencyContainer using
    // Kernel::makeResident which walks all of a kernel's allocations
    // (privateSurface + program constant/global/exportedFunctions +
    // kernelSvmGfxAllocations + kernelUnifiedMemoryGfxAllocations +
    // bindless heaps + per-arg cl_mem GraphicsAllocations). This
    // accumulates into the CSR's internal residency list. We then
    // additionally push our heaps + commandBufferAllocation directly
    // into the explicit container passed to submitBatchBuffer.
    auto &gpgpuCsr2 = queue->getGpgpuCommandStreamReceiver();
    auto rootDeviceIndex = queue->getDevice().getRootDeviceIndex();
    for (const auto &rec : recordedCommands) {
        if (rec.kernelClone == nullptr) continue;
        Kernel *pKernel = rec.kernelClone->getKernel(rootDeviceIndex);
        if (pKernel == nullptr) continue;
        // Kernel::makeResident enumerates and registers every allocation
        // the kernel touches with the CSR -- canonical residency setup
        // mirroring NEO's regular enqueue flow.
        pKernel->makeResident(gpgpuCsr2);
    }

    ResidencyContainer residencyContainer;

    // The persistent batch buffer itself (commands the GPU executes)
    residencyContainer.push_back(commandBufferAllocation);

    // KernelOperation's indirect heaps populated by dispatchWalker.
    // These hold INTERFACE_DESCRIPTOR_DATA, surface state, dynamic state,
    // sampler state -- all addressed by the recorded GPGPU walker commands.
    auto pushHeapAlloc = [&](IndirectHeap *heap) {
        if (heap == nullptr) return;
        if (auto *ga = heap->getGraphicsAllocation()) {
            residencyContainer.push_back(ga);
        }
    };
    pushHeapAlloc(deepCsrKernelOp->dsh.get());
    pushHeapAlloc(deepCsrKernelOp->ioh.get());
    pushHeapAlloc(deepCsrKernelOp->ssh.get());

    // Every recorded kernel's binary GraphicsAllocation (also covered
    // by Kernel::makeResident above via program lookup, but pushed
    // explicitly to be safe in case the program path doesn't track
    // the kernel ISA allocation).
    for (const auto &rec : recordedCommands) {
        if (rec.kernelClone == nullptr) continue;
        Kernel *pKernel = rec.kernelClone->getKernel(rootDeviceIndex);
        if (pKernel == nullptr) continue;
        const auto &kernelInfo = pKernel->getKernelInfo();
        if (auto *kernelAllocation = kernelInfo.getGraphicsAllocation()) {
            residencyContainer.push_back(kernelAllocation);
        }
    }

    batchBuffer.allocationsForResidency = &residencyContainer;

    auto &gpgpuCsr = queue->getGpgpuCommandStreamReceiver();
    auto submissionStatus = gpgpuCsr.submitBatchBuffer(batchBuffer,
                                                        residencyContainer);
    if (submissionStatus != SubmissionStatus::success) {
        return CL_INVALID_OPERATION;
    }

    // Sprint 1H TODO: resolve cl_event handle with proper task count
    // tracking (event needs to signal completion when the submitted
    // batch buffer finishes on GPU). For now don't expose an event --
    // caller-provided event will not signal correctly until 1H.
    return CL_SUCCESS;
}

} // namespace NEO

// =============================================================================
// API entry points
// =============================================================================

extern "C" {

CL_API_ENTRY cl_command_buffer_khr CL_API_CALL
clCreateCommandBufferKHR(cl_uint numQueues,
                         const cl_command_queue *queues,
                         const cl_command_buffer_properties_khr *properties,
                         cl_int *errcodeRet) {
    if (numQueues != 1 || queues == nullptr || queues[0] == nullptr) {
        if (errcodeRet) *errcodeRet = CL_INVALID_VALUE;
        return nullptr;
    }
    auto neoQueue = NEO::castToObject<NEO::CommandQueue>(queues[0]);
    if (neoQueue == nullptr) {
        if (errcodeRet) *errcodeRet = CL_INVALID_COMMAND_QUEUE;
        return nullptr;
    }
    auto cmdBuffer = new NEO::ClCommandBuffer(neoQueue, properties);
    if (errcodeRet) *errcodeRet = CL_SUCCESS;
    return NEO::toClHandle(cmdBuffer);
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainCommandBufferKHR(cl_command_buffer_khr commandBuffer) {
    auto cb = NEO::toNeoCommandBuffer(commandBuffer);
    if (cb == nullptr) return CL_INVALID_COMMAND_BUFFER_KHR;
    cb->retain();
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseCommandBufferKHR(cl_command_buffer_khr commandBuffer) {
    auto cb = NEO::toNeoCommandBuffer(commandBuffer);
    if (cb == nullptr) return CL_INVALID_COMMAND_BUFFER_KHR;
    if (cb->release()) {
        delete cb;
    }
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clFinalizeCommandBufferKHR(cl_command_buffer_khr commandBuffer) {
    auto cb = NEO::toNeoCommandBuffer(commandBuffer);
    if (cb == nullptr) return CL_INVALID_COMMAND_BUFFER_KHR;
    return cb->finalize();
}

CL_API_ENTRY cl_int CL_API_CALL
clCommandNDRangeKernelKHR(cl_command_buffer_khr commandBuffer,
                          cl_command_queue commandQueue,
                          const cl_command_properties_khr * /*properties*/,
                          cl_kernel kernel,
                          cl_uint workDim,
                          const size_t *globalWorkOffset,
                          const size_t *globalWorkSize,
                          const size_t *localWorkSize,
                          cl_uint /*numSyncPointsInWaitList*/,
                          const cl_sync_point_khr * /*syncPointWaitList*/,
                          cl_sync_point_khr * /*syncPoint*/,
                          cl_mutable_command_khr *mutableHandle) {
    auto cb = NEO::toNeoCommandBuffer(commandBuffer);
    if (cb == nullptr) return CL_INVALID_COMMAND_BUFFER_KHR;
    auto queue = NEO::castToObject<NEO::CommandQueue>(commandQueue);
    return cb->recordNDRangeKernel(queue, kernel, workDim,
                                    globalWorkOffset, globalWorkSize, localWorkSize,
                                    mutableHandle);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCommandBufferKHR(cl_uint numQueues,
                          cl_command_queue *queues,
                          cl_command_buffer_khr commandBuffer,
                          cl_uint numEventsInWaitList,
                          const cl_event *eventWaitList,
                          cl_event *event) {
    auto cb = NEO::toNeoCommandBuffer(commandBuffer);
    if (cb == nullptr) return CL_INVALID_COMMAND_BUFFER_KHR;
    NEO::CommandQueue *queue = nullptr;
    if (numQueues != 0) {
        if (numQueues != 1 || queues == nullptr || queues[0] == nullptr) {
            return CL_INVALID_VALUE;
        }
        queue = NEO::castToObject<NEO::CommandQueue>(queues[0]);
    }
    return cb->enqueue(queue, numEventsInWaitList, eventWaitList, event);
}

CL_API_ENTRY cl_int CL_API_CALL
clGetCommandBufferInfoKHR(cl_command_buffer_khr commandBuffer,
                          cl_command_buffer_info_khr paramName,
                          size_t paramValueSize,
                          void *paramValue,
                          size_t *paramValueSizeRet) {
    auto cb = NEO::toNeoCommandBuffer(commandBuffer);
    if (cb == nullptr) return CL_INVALID_COMMAND_BUFFER_KHR;
    // Minimal info support: state + queue list (subset of spec).
    switch (paramName) {
    case CL_COMMAND_BUFFER_STATE_KHR: {
        cl_command_buffer_state_khr s = CL_COMMAND_BUFFER_STATE_PENDING_KHR;
        switch (cb->getState()) {
        case NEO::ClCommandBufferState::recording:       s = CL_COMMAND_BUFFER_STATE_RECORDING_KHR; break;
        case NEO::ClCommandBufferState::executable:      s = CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR; break;
        case NEO::ClCommandBufferState::pending:
        case NEO::ClCommandBufferState::invalidInternal: s = CL_COMMAND_BUFFER_STATE_PENDING_KHR; break;
        }
        if (paramValueSizeRet) *paramValueSizeRet = sizeof(s);
        if (paramValue) {
            if (paramValueSize < sizeof(s)) return CL_INVALID_VALUE;
            memcpy(paramValue, &s, sizeof(s));
        }
        return CL_SUCCESS;
    }
    default:
        return CL_INVALID_VALUE;
    }
}

CL_API_ENTRY cl_int CL_API_CALL
clUpdateMutableCommandsKHR(cl_command_buffer_khr commandBuffer,
                            cl_uint numConfigs,
                            const cl_command_buffer_update_type_khr *configTypes,
                            const void **configs) {
    auto cb = NEO::toNeoCommandBuffer(commandBuffer);
    if (cb == nullptr) return CL_INVALID_COMMAND_BUFFER_KHR;
    return cb->updateMutableCommands(numConfigs, configTypes, configs);
}

} // extern "C"
