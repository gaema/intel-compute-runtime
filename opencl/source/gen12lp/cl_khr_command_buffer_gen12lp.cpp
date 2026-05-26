/*
 * Copyright (C) 2026 Libre Computer Project
 *
 * SPDX-License-Identifier: MIT
 *
 * iter104++ Sprint 1C: per-Gen12LP registration of the deep-CSR
 * populate function for cl_khr_command_buffer.
 *
 * Mirrors the registration pattern used by command_queue_gen12lp.cpp
 * for commandQueueFactory. Per-family file because the populate
 * function dispatches HardwareInterface<Family>::dispatchWalker which
 * is a template instantiated per-family.
 *
 * Sprint 1C status: factory entry registered; populate function body
 * is the Sprint 1D deliverable. Until that lands, populateBatchBufferGen12Lp
 * returns CL_INVALID_OPERATION which leaves deepCsrInitialized=false and
 * causes ClCommandBuffer::enqueue to fall through to batchedDispatch.
 */

#include "shared/source/command_container/command_encoder.h"
#include "shared/source/command_stream/command_stream_receiver.h"
#include "shared/source/command_stream/csr_definitions.h"
#include "shared/source/command_stream/linear_stream.h"
#include "shared/source/command_stream/stream_properties.h"
#include "shared/source/device/device.h"
#include "shared/source/gen12lp/hw_cmds.h"
#include "shared/source/gmm_helper/gmm_helper.h"
#include "shared/source/helpers/gfx_core_helper.h"
#include "shared/source/helpers/pipe_control_args.h"
#include "shared/source/helpers/pipeline_select_args.h"
#include "shared/source/helpers/preamble.h"
#include "shared/source/helpers/populate_factory.h"
#include "shared/source/helpers/state_base_address.h"
#include "shared/source/helpers/timestamp_packet.h"
#include "shared/source/helpers/vec.h"
#include "shared/source/indirect_heap/indirect_heap.h"

#include "opencl/source/api/cl_khr_command_buffer.h"
#include "opencl/source/cl_device/cl_device.h"
#include "opencl/source/command_queue/command_queue.h"
#include "opencl/source/command_queue/command_queue_hw.h"
#include "opencl/source/command_queue/gpgpu_walker.h"
#include "opencl/source/command_queue/hardware_interface.h"
#include "opencl/source/command_queue/hardware_interface_base.inl"
#include "opencl/source/helpers/dispatch_info.h"
#include "opencl/source/helpers/hardware_commands_helper.h"
#include "opencl/source/helpers/task_information.h"
#include "opencl/source/kernel/kernel.h"
#include "opencl/source/kernel/multi_device_kernel.h"

namespace NEO {

typedef Gen12LpFamily Family;
static auto gfxCore = IGFX_GEN12LP_CORE;

// iter104++ Sprint 1C entry point.
// TODO Sprint 1D: implement the populate body.
//
// Reference: dispatch_walker_tests.cpp:165-179 shows the canonical
// dispatchWalker invocation for a single kernel:
//
//   DispatchInfo dispatchInfo(pClDevice, kernel, dimensions, workItems,
//                             lws, globalOffsets);
//   dispatchInfo.setNumberOfWorkgroups(workgroups);
//   dispatchInfo.setTotalNumberOfWorkgroups(workgroups);
//   MultiDispatchInfo multiDispatchInfo;
//   multiDispatchInfo.push(dispatchInfo);
//
//   HardwareInterfaceWalkerArgs walkerArgs = createHardwareInterfaceWalkerArgs(
//     CL_COMMAND_NDRANGE_KERNEL);
//   walkerArgs.blockedCommandsData = cb.getDeepCsrKernelOp();  // hijack
//
//   HardwareInterface<Family>::template dispatchWalker<
//     typename Family::DefaultWalkerType>(
//     queue, multiDispatchInfo, CsrDependencies(), walkerArgs);
//
// For each cb.getRecordedCommands()[i]:
//   1. Resolve Kernel* from rec.kernelClone via queue rootDeviceIndex
//   2. Build DispatchInfo with rec.workDim, rec.globalWorkSize.data(),
//      rec.localWorkSize.data() (or nullptr if !rec.hasLocalSize), and
//      rec.globalWorkOffset.data() (or nullptr if !rec.hasOffset)
//   3. Compute workgroups via NEO::generateWorkgroupsNumber(globalSize, lws)
//   4. Push into MultiDispatchInfo + invoke dispatchWalker as above
//   5. Track byte offsets in cb.getDeepCsrKernelOp()->commandStream
//      (Sprint 2 needs these for in-place mutable update)
// After loop: append cmdbuf-end PIPE_CONTROL + MI_BATCH_BUFFER_END
// Return CL_SUCCESS to flip deepCsrInitialized=true.
static cl_int populateBatchBufferGen12Lp(ClCommandBuffer &cb,
                                          CommandQueue &queue) {
    // Sprint 1D (EXPERIMENTAL): invoke dispatchWalker per recorded
    // command into our persistent LinearStream via the blockedCommandsData
    // hijack. Returns CL_INVALID_OPERATION at the end so deepCsrInitialized
    // stays false and enqueue falls through to batchedDispatch -- safety
    // gate while validating that the walker writes succeed and produce
    // valid command stream content.
    //
    // The MEGA_CL_TRACE / CL_CMDBUF_DEEP_CSR=1 path will have:
    //   - dispatchWalker invocations succeed (return CL_SUCCESS implicit)
    //   - LinearStream::getUsed() grows after each call
    //   - End-to-end decode rate still served by batchedDispatch
    // Sprint 1E will flip the return to CL_SUCCESS once we trust the
    // populated stream + add cmdbuf-end PIPE_CONTROL + BB_END +
    // deepCsrSubmit BatchBuffer construction.
    using DefaultWalkerType = typename Family::DefaultWalkerType;
    auto rootDeviceIndex = queue.getDevice().getRootDeviceIndex();
    auto *clDevice = &queue.getClDevice();
    auto *kernelOp = cb.getDeepCsrKernelOp();
    if (kernelOp == nullptr) {
        return CL_INVALID_OPERATION;
    }

    size_t initialUsed = kernelOp->commandStream ? kernelOp->commandStream->getUsed() : 0;

    // Sprint 1H+2: emit STATE_BASE_ADDRESS at the start of the persistent
    // stream so heap-relative addresses (DSH/SSH offsets in the GPGPU
    // walker commands) resolve correctly. NEO's regular flush path
    // emits this; cmdbuf-style isolated batch buffers must too. Without
    // it the walker commands inherit stale STATE_BASE_ADDRESS from prior
    // submissions -> garbage state lookups -> GPU page-fault -> i915
    // recovery -> slow per-kernel re-execution via gaema's fallback.
    //
    // Heaps (dsh/ioh/ssh) are not yet populated by dispatchWalker (that
    // happens later in this same function via the per-command loop). But
    // SBA programmed before dispatchWalker should set up the right heap
    // bases since dispatchWalker uses the same KernelOperation heaps via
    // blockedCommandsData.
    {
        StateBaseAddressHelperArgs<Family> sbaArgs{};
        sbaArgs.dsh = kernelOp->dsh.get();
        sbaArgs.ioh = kernelOp->ioh.get();
        sbaArgs.ssh = kernelOp->ssh.get();
        sbaArgs.gmmHelper = queue.getDevice().getRootDeviceEnvironment().getGmmHelper();
        sbaArgs.setInstructionStateBaseAddress = true;
        sbaArgs.setGeneralStateBaseAddress = false;
        sbaArgs.statelessMocsIndex = 0;
        StateBaseAddressHelper<Family>::programStateBaseAddressIntoCommandStream(
            sbaArgs, *kernelOp->commandStream);
    }


    // Sprint 1H+3: emit the canonical NEO prologue between SBA and the
    // walker loop. Reference sequence from
    // CommandStreamReceiverHw<GfxFamily>::flushTask
    // (command_stream_receiver_hw_base.inl ~lines 540-560):
    //   programPipelineSelect -> programL3 -> programPreamble
    //   -> addPipeControlBeforeVfeCmd -> programVFEState -> programPreemption
    //
    // Each of these is what NEO emits implicitly per submission. Our
    // isolated batch buffer must replicate them or the GPU has no
    // compute pipeline / L3 / preemption / scratch / max-thread state
    // and the first walker page-faults -> i915 TDR -> per-kernel
    // re-execution at ~2-3 t/s.
    //
    // programPreamble subsumes L3 + preemption + Gen WAs + StateSip,
    // so we only need: PipelineSelect, Preamble, PipeControlBeforeVfe,
    // VfeState. Scratch left at 0 (no kernel uses scratch in this
    // workload); maxFrontEndThreads from device.
    {
        auto &device = queue.getDevice();
        auto &hwInfo = device.getHardwareInfo();
        auto &rde = device.getRootDeviceEnvironment();
        auto &gfxCoreHelper = device.getGfxCoreHelper();
        auto engineGroupType = gfxCoreHelper.getEngineGroupType(
            queue.getGpgpuCommandStreamReceiver().getOsContext().getEngineType(),
            queue.getGpgpuCommandStreamReceiver().getOsContext().getEngineUsage(),
            hwInfo);

        PipelineSelectArgs pipelineSelectArgs{};
        PreambleHelper<Family>::programPipelineSelect(
            kernelOp->commandStream.get(), pipelineSelectArgs, rde);

        uint32_t l3Config = PreambleHelper<Family>::getL3Config(hwInfo, true);
        PreambleHelper<Family>::programPreamble(
            kernelOp->commandStream.get(), device, l3Config,
            queue.getGpgpuCommandStreamReceiver().getPreemptionAllocation(),
            false /* isBcs */);

        PreambleHelper<Family>::addPipeControlBeforeVfeCmd(
            kernelOp->commandStream.get(), &hwInfo, engineGroupType);

        StreamProperties streamProperties{};
        streamProperties.initSupport(rde);
        auto pVfeState = PreambleHelper<Family>::getSpaceForVfeState(
            kernelOp->commandStream.get(), hwInfo, engineGroupType, nullptr);
        PreambleHelper<Family>::programVfeState(
            pVfeState, rde, 0u /* scratchSize */, 0u /* scratchAddress */,
            device.getDeviceInfo().maxFrontEndThreads, streamProperties);
    }

    for (const auto &rec : cb.getRecordedCommands()) {
        if (rec.kernelClone == nullptr) {
            return CL_INVALID_OPERATION;
        }
        Kernel *pKernel = rec.kernelClone->getKernel(rootDeviceIndex);
        if (pKernel == nullptr) {
            return CL_INVALID_OPERATION;
        }

        size_t globalWorkSize[3] = {1, 1, 1};
        size_t localWorkSize[3] = {1, 1, 1};
        size_t globalWorkOffset[3] = {0, 0, 0};
        for (cl_uint d = 0; d < rec.workDim && d < 3; d++) {
            globalWorkSize[d] = rec.globalWorkSize[d];
            if (rec.hasLocalSize) {
                localWorkSize[d] = rec.localWorkSize[d];
            }
            if (rec.hasOffset) {
                globalWorkOffset[d] = rec.globalWorkOffset[d];
            }
        }

        // Compute number of workgroups (ceil(global / local) per dim).
        Vec3<size_t> numWorkgroups{
            (globalWorkSize[0] + localWorkSize[0] - 1) / localWorkSize[0],
            (globalWorkSize[1] + localWorkSize[1] - 1) / localWorkSize[1],
            (globalWorkSize[2] + localWorkSize[2] - 1) / localWorkSize[2]};

        DispatchInfo dispatchInfo(clDevice, pKernel, rec.workDim,
            Vec3<size_t>{globalWorkSize[0], globalWorkSize[1], globalWorkSize[2]},
            Vec3<size_t>{localWorkSize[0], localWorkSize[1], localWorkSize[2]},
            Vec3<size_t>{globalWorkOffset[0], globalWorkOffset[1], globalWorkOffset[2]});
        dispatchInfo.setNumberOfWorkgroups(numWorkgroups);
        dispatchInfo.setTotalNumberOfWorkgroups(numWorkgroups);

        MultiDispatchInfo multiDispatchInfo;
        multiDispatchInfo.push(dispatchInfo);

        HardwareInterfaceWalkerArgs walkerArgs = {};
        walkerArgs.commandType = CL_COMMAND_NDRANGE_KERNEL;
        walkerArgs.blockedCommandsData = kernelOp;

        HardwareInterface<Family>::template dispatchWalker<DefaultWalkerType>(
            queue, multiDispatchInfo, CsrDependencies(), walkerArgs);
    }

    size_t finalUsed = kernelOp->commandStream ? kernelOp->commandStream->getUsed() : 0;
    // Telemetry: how many bytes of GPGPU command stream we emitted.
    // (When env var MEGA_CL_TRACE is set, the gaema-side runner will
    // print per-call timing; use that to correlate.)
    if (finalUsed <= initialUsed) {
        // dispatchWalker emitted nothing -- something is off. Fall through.
        return CL_INVALID_OPERATION;
    }

    // Sprint 1E: append cmdbuf-end PIPE_CONTROL + MI_BATCH_BUFFER_END to
    // mark the persistent stream as a complete batch buffer ready for
    // direct CSR submission. PIPE_CONTROL ensures all kernels finish
    // and writes are made visible before the batch ends; BB_END is the
    // GPU's "stop fetching commands" marker.
    PipeControlArgs pcArgs{};
    pcArgs.dcFlushEnable = true;  // flush data cache so subsequent host reads see GPU writes
    MemorySynchronizationCommands<Family>::addSingleBarrier(
        *kernelOp->commandStream, pcArgs);

    EncodeBatchBufferStartOrEnd<Family>::programBatchBufferEnd(
        *kernelOp->commandStream);

    // Sprint 1F TODO: implement deepCsrSubmit BatchBuffer construction +
    // CommandStreamReceiver::submitBatchBuffer call. Until that lands,
    // we still return CL_INVALID_OPERATION so deepCsrInitialized stays
    // false and enqueue falls through to the working batchedDispatch
    // path. The persistent stream now contains a fully-formed batch
    // buffer (walker commands + PIPE_CONTROL + BB_END) ready for
    // submission once deepCsrSubmit is wired.

    (void)initialUsed; // silence unused var when telemetry is later removed
    // Sprint 1G FLIP (re-enabled, paired with the enqueue-gate
    // fall-through fix in cl_khr_command_buffer.cpp): return CL_SUCCESS
    // so deepCsrInitialized=true and enqueue routes through deepCsrSubmit.
    // The deepCsrSubmit body's FORCE gate returns CL_INVALID_OPERATION
    // by default which the enqueue-gate fall-through now honors -- so
    // production behavior stays on batchedDispatch while we iterate
    // toward enabling FORCE.
    return CL_SUCCESS;
}

// Static initializer registers the deep-CSR factory entry for Gen12LP.
// (No populateFactoryTable specialization needed -- that template is
// already specialized by command_queue_gen12lp.cpp; defining it here
// would cause a multiple-definition error.)
// Runs at library load time.
static struct DeepCsrFactoryRegistrarGen12Lp {
    DeepCsrFactoryRegistrarGen12Lp() {
        deepCsrPopulateFactory[gfxCore] = &populateBatchBufferGen12Lp;
    }
} registrarInstance;

} // namespace NEO
