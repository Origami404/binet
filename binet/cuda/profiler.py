"""Replace the first matching launch and deliver one completed Torch trace buffer."""
from dataclasses import dataclass
import hashlib
import math
import os
import threading

from binet.cuda.core.cubin import Cubin
from binet.cuda.hook import Callbacks
from binet.records import RECORD_BYTES, geometry


def _checked(result):
    status, *values = result
    if int(status):
        raise RuntimeError(f"CUDA driver call failed: {status!r}")
    return values[0] if len(values) == 1 else tuple(values)


@dataclass
class Capture:
    """Owned, completed CPU uint8 tensor and the captured launch's geometry."""
    data: object
    context: int
    grid: tuple
    block: tuple


class CUDAProfiler:
    """Profile the first matching launch; call on_capture(Capture) after completion.

    Torch owns a fresh device buffer and its pinned CPU snapshot. The captured
    launch is synchronized; later launches run normally. Only primary contexts
    are supported. Each profiler instance attempts at most one matching launch.
    Source modules must load while subscribed. Callback/cleanup failures remain
    in errors.
    """
    def __init__(self, injected, on_capture):
        self.injected, self.on_capture = injected, on_capture
        self._injected_sha = hashlib.sha256(injected.image).hexdigest()
        self._param_base = Cubin(injected.image).kernel(injected.kernel).param_base
        self._modules, self._images = {}, {}
        self._library = None
        self._lock, self._launch_lock = threading.RLock(), threading.Lock()
        self._started = self._stopping = self._claimed = False
        self._hooks = Callbacks(on_module_load=self._module_load, on_module_unload=self._module_unload,
                                on_context_destroy=self._context_destroy, on_launch_enter=self._enter,
                                on_launch_exit=self._exit, on_graph_launch=self._graph)
        self.errors = self._hooks.errors

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        if self._started:
            return
        import torch
        from cuda.bindings import driver
        self.torch, self.cu = torch, driver
        # Initialize Torch outside a native callback; later allocations must not
        # recursively initialize its CUDA runtime while a driver launch is paused.
        torch.cuda.init()
        self._stopping = False
        self._hooks.start()
        self._started = True

    def stop(self):
        if not self._started:
            return
        self._stopping = True
        with self._launch_lock:
            pass
        self._hooks.stop()
        self._started = False
        self._modules.clear()
        self._images.clear()
        if self._library is not None:
            try:
                _checked(self.cu.cuLibraryUnload(self._library))
            except Exception as error:
                self.errors.append(f"profile cleanup: {error}")
            finally:
                self._library = None

    def _module_load(self, context, module_id, image):
        if self._claimed:
            return
        digest = hashlib.sha256(image).hexdigest()
        if digest == self._injected_sha:
            return
        with self._lock:
            if digest not in self._images:
                self._images[digest] = (digest == self.injected.source_cubin_sha256
                                        or self.injected.kernel in Cubin(image).kernels)
            if self._images[digest]:
                self._modules[(context, module_id)] = digest

    def _module_unload(self, context, module_id):
        with self._lock:
            self._modules.pop((context, module_id), None)

    def _context_destroy(self, context):
        with self._lock:
            self._modules = {key: value for key, value in self._modules.items() if key[0] != context}

    def _source(self, context):
        with self._lock:
            found = {digest for (ctx, _), digest in self._modules.items() if ctx in (0, context)}
        if not found:
            raise RuntimeError(f"no loaded source cubin for {self.injected.kernel!r}")
        if len(found) != 1:
            raise RuntimeError(f"multiple live cubins define {self.injected.kernel!r}")
        if found.pop() != self.injected.source_cubin_sha256:
            raise RuntimeError(f"source cubin SHA256 differs from injected artifact for {self.injected.kernel!r}")

    def _graph(self):
        if not self._claimed:
            raise RuntimeError("CUDA graph launches are not expanded or instrumented")

    def _enter(self, launch):
        if launch.name != self.injected.kernel or self._claimed:
            return None
        self._launch_lock.acquire()
        try:
            if self._stopping or self._claimed or self.errors:
                self._launch_lock.release()
                return None
            self._claimed = True
            context = int(_checked(self.cu.cuCtxGetCurrent()))
            if not context or launch.context not in (0, context):
                raise RuntimeError("matching launch has no consistent current CUDA context")
            self._source(context)
            if int(_checked(self.cu.cuStreamIsCapturing(self.cu.CUstream(launch.stream)))):
                raise RuntimeError("profiling launches during CUDA graph capture is not supported")
            grid, block, warps = geometry(launch.grid, launch.block)
            size = warps * self.injected.slot_bytes
            limit = int(os.environ.get("BINET_PROFILE_BUFFER_MIB", "1024"))
            if limit <= 0 or size > limit * (1 << 20):
                raise RuntimeError(f"trace needs {size} bytes; raise positive BINET_PROFILE_BUFFER_MIB or reduce ring depth")
            device = int(_checked(self.cu.cuCtxGetDevice()))
            primary = _checked(self.cu.cuDevicePrimaryCtxRetain(self.cu.CUdevice(device)))
            try:
                if int(primary) != context:
                    raise RuntimeError("Torch trace buffers require a CUDA primary context")
            finally:
                _checked(self.cu.cuDevicePrimaryCtxRelease(self.cu.CUdevice(device)))
            function, layout = self._prepare(launch.function)
            arguments = launch.arguments(layout)
            stream = self.torch.cuda.ExternalStream(launch.stream, device=device)
            shape = (math.prod(grid), (math.prod(block) + 31) // 32, self.injected.ring_depth, RECORD_BYTES)
            with self.torch.cuda.device(device), self.torch.cuda.stream(stream):
                # Zero on the intercepted stream, ordered before the original launch.
                buffer = self.torch.zeros(shape, dtype=self.torch.uint8, device=device)
            launch.replace(function, arguments, buffer.data_ptr())
            return dict(context=context, grid=grid, block=block, stream=stream, buffer=buffer)
        except BaseException:
            self._launch_lock.release()
            raise

    def _exit(self, launch, active):
        try:
            if launch.status != 0:
                raise RuntimeError(f"{launch.api} failed with CUDA result {launch.status}")
            stream, buffer = active["stream"], active["buffer"]
            stream.synchronize()
            host = self.torch.empty(buffer.shape, dtype=self.torch.uint8, device="cpu", pin_memory=True)
            with self.torch.cuda.stream(stream):
                host.copy_(buffer)  # Blocking D2H; host is complete before the handler runs.
            self.on_capture(Capture(host, active["context"], active["grid"], active["block"]))
        finally:
            self._launch_lock.release()

    def _prepare(self, incoming):
        cu = self.cu
        original = cu.CUfunction(incoming)
        first = cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        if int(cu.cuFuncGetAttribute(first, original)[0]):
            original = _checked(cu.cuKernelGetFunction(cu.CUkernel(incoming)))
        self._library = _checked(cu.cuLibraryLoadData(self.injected.image, [], [], 0, [], [], 0))
        kernel = _checked(cu.cuLibraryGetKernel(self._library, self.injected.kernel.encode()))
        replacement = _checked(cu.cuKernelGetFunction(kernel))
        layout = []
        for i in range(self.injected.param_count):
            old = tuple(map(int, _checked(cu.cuFuncGetParamInfo(original, i))))
            new = tuple(map(int, _checked(cu.cuFuncGetParamInfo(replacement, i))))
            if old != new:
                raise RuntimeError(f"rewritten parameter {i} layout differs from original")
            layout.append(old)
        appended = tuple(map(int, _checked(cu.cuFuncGetParamInfo(replacement, self.injected.param_count))))
        if appended != (self.injected.param_offset - self._param_base, 8):
            raise RuntimeError("rewritten buffer parameter layout differs from artifact")
        for name in ("MAX_DYNAMIC_SHARED_SIZE_BYTES", "PREFERRED_SHARED_MEMORY_CARVEOUT",
                     "NON_PORTABLE_CLUSTER_SIZE_ALLOWED", "CLUSTER_SCHEDULING_POLICY_PREFERENCE"):
            attr = getattr(cu.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + name)
            status, value = cu.cuFuncGetAttribute(attr, original)
            if not int(status):
                _checked(cu.cuFuncSetAttribute(replacement, attr, value))
        return replacement, layout
