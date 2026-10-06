"""Scoped CUPTI callbacks and owned module/launch data."""
import ctypes as C
from contextlib import contextmanager
import threading


class ModuleResource(C.Structure):
    # CUpti_ModuleResourceData, cupti_callbacks.h; cubins contain embedded NULs.
    _fields_ = [("module_id", C.c_uint32), ("cubin_size", C.c_size_t), ("p_cubin", C.c_void_p)]


class LaunchConfig(C.Structure):
    # Public CUlaunchConfig ABI. Leave launch attributes untouched.
    _fields_ = [("gx", C.c_uint), ("gy", C.c_uint), ("gz", C.c_uint),
                ("bx", C.c_uint), ("by", C.c_uint), ("bz", C.c_uint),
                ("shared", C.c_uint), ("stream", C.c_void_p),
                ("attrs", C.c_void_p), ("num_attrs", C.c_uint)]


class Launch:
    """One driver launch. Parameter storage and replacements live through API_EXIT."""
    def __init__(self, data, params_type):
        self._owner, self._params_type = data, params_type
        # owner prevents cupti-python from copying the native parameter structure.
        self._params = params_type.from_ptr(int(data.function_params), owner=data)
        params = self._params
        self.name, self.api = data.symbol_name or "", data.function_name
        self.context, self.function = int(data.context), int(params.f)
        self.status = None
        self._saved = (self.function, int(params.kernel_params), int(getattr(params, "extra", 0)))
        self._pointer = self._argv = None
        if hasattr(params, "config"):
            if not params.config:
                raise RuntimeError("null CUlaunchConfig")
            config = LaunchConfig.from_address(int(params.config))
            self.grid, self.block = (config.gx, config.gy, config.gz), (config.bx, config.by, config.bz)
            self.stream = int(config.stream or 0)
        else:
            self.grid = (int(params.grid_dim_x), int(params.grid_dim_y), int(params.grid_dim_z))
            self.block = (int(params.block_dim_x), int(params.block_dim_y), int(params.block_dim_z))
            self.stream = int(params.h_stream)
        if not self.stream and self.api.endswith("_ptsz"):
            self.stream = 2  # CU_STREAM_PER_THREAD

    def arguments(self, layout):
        """Pointers into the caller's arguments, for kernelParams or packed extra."""
        argv, extra = self._saved[1:]
        if argv and extra:
            raise RuntimeError("both kernelParams and extra were supplied")
        if argv:
            pointers = list((C.c_void_p * len(layout)).from_address(argv))
            if any(pointer is None for pointer in pointers):
                raise RuntimeError("null kernel argument pointer")
            return pointers
        packed, size = 0, 0
        if extra:
            items, seen = C.cast(extra, C.POINTER(C.c_void_p)), set()
            for i in range(0, 16, 2):
                token = items[i]
                if not token:
                    break
                if token not in (1, 2) or token in seen or not items[i + 1]:
                    raise RuntimeError("unsupported launch extra parameter")
                seen.add(token)
                if token == 1:
                    packed = items[i + 1]
                else:
                    size = C.c_size_t.from_address(items[i + 1]).value
            else:
                raise RuntimeError("unterminated launch extra parameters")
            if seen != {1, 2}:
                raise RuntimeError("missing packed launch buffer or size")
        if any(not packed or offset + length > size for offset, length in layout):
            raise RuntimeError("missing kernel argument bytes")
        return [packed + offset for offset, _ in layout]

    def replace(self, function, arguments, buffer):
        """Substitute a function and append its trace-buffer pointer; restored by the hook."""
        self._pointer = C.c_uint64(buffer)
        self._argv = (C.c_void_p * (len(arguments) + 1))(*arguments, C.addressof(self._pointer))
        self._params.f = int(function)
        self._params.kernel_params = C.addressof(self._argv)
        if hasattr(self._params, "extra"):
            self._params.extra = 0
        alias = self._params_type.from_ptr(int(self._owner.function_params), owner=self._owner)
        if int(alias.f) != int(function) or int(alias.kernel_params) != C.addressof(self._argv):
            raise RuntimeError("CUPTI launch parameter writes did not reach callback storage")

    def _restore(self):
        self._params.f, self._params.kernel_params = self._saved[:2]
        if hasattr(self._params, "extra"):
            self._params.extra = self._saved[2]


class Callbacks:
    """One CUPTI subscription, including reentrancy and launch pairing.

    on_launch_enter returns state for on_launch_exit(launch, state), or None to
    ignore a launch. Paired launches are restored even if a handler fails.
    stop() retains errors; context-manager exit raises them only when the body
    succeeded. Module callbacks receive owned bytes, never borrowed cubin memory.
    """
    def __init__(self, *, on_module_load=None, on_module_unload=None, on_context_destroy=None,
                 on_launch_enter=None, on_launch_exit=None, on_graph_launch=None):
        if (on_launch_enter is None) != (on_launch_exit is None):
            raise ValueError("launch enter and exit handlers must be supplied together")
        self.on_module_load, self.on_module_unload = on_module_load, on_module_unload
        self.on_context_destroy = on_context_destroy
        self.on_launch_enter, self.on_launch_exit = on_launch_enter, on_launch_exit
        self.on_graph_launch = on_graph_launch
        self.errors = []
        self._subscriber = None
        self._local, self._lock = threading.local(), threading.Lock()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        if exc_type is None and self.errors:
            raise RuntimeError("CUPTI callback failed: " + self.errors[0])

    def start(self):
        if self._subscriber is not None:
            return
        from cupti import cupti
        self.cupti = cupti
        self._params, self._graphs = {}, set()
        if self.on_launch_enter is not None:
            for api, typename in (("cuLaunchKernel", "CuLaunchKernelParams"),
                                  ("cuLaunchKernelEx", "CuLaunchKernelExParams"),
                                  ("cuLaunchCooperativeKernel", "CuLaunchCooperativeKernelParams")):
                for suffix in ("", "_ptsz"):
                    self._params[int(getattr(cupti.DriverApiTraceCbid, api + suffix))] = getattr(cupti, typename)
        if self.on_graph_launch is not None:
            self._graphs = {int(getattr(cupti.DriverApiTraceCbid, name))
                            for name in ("cuGraphLaunch", "cuGraphLaunch_ptsz")}
        self._subscriber = cupti.subscribe(self._callback, None)
        try:
            for name, handler in (("MODULE_LOADED", self.on_module_load),
                                  ("MODULE_UNLOAD_STARTING", self.on_module_unload),
                                  ("CONTEXT_DESTROY_STARTING", self.on_context_destroy)):
                if handler is not None:
                    cupti.enable_callback(1, self._subscriber, cupti.CallbackDomain.RESOURCE,
                                          getattr(cupti.CallbackIdResource, name))
            for cbid in self._params.keys() | self._graphs:
                cupti.enable_callback(1, self._subscriber, cupti.CallbackDomain.DRIVER_API, cbid)
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self._subscriber is not None:
            self.cupti.unsubscribe(self._subscriber)
            self._subscriber = None

    def _callback(self, userdata, domain, cbid, data):
        if getattr(self._local, "inside", False):
            return
        self._local.inside = True
        try:
            if int(domain) == int(self.cupti.CallbackDomain.RESOURCE):
                if not hasattr(data, "resource_descriptor"):
                    data = self.cupti.ResourceData.from_ptr(int(data), owner=self)
                self._resource(int(cbid), data)
            elif int(domain) == int(self.cupti.CallbackDomain.DRIVER_API):
                if not hasattr(data, "function_params"):
                    data = self.cupti.CallbackData.from_ptr(int(data), owner=self)
                self._launch(int(cbid), data)
        except BaseException as error:
            with self._lock:
                message = f"{type(error).__name__}: {error}"
                if message not in self.errors:
                    self.errors.append(message)
        finally:
            self._local.inside = False

    def _resource(self, cbid, data):
        ids, context = self.cupti.CallbackIdResource, int(data.context)
        if cbid == ids.CONTEXT_DESTROY_STARTING:
            self.on_context_destroy(context)
        elif data.resource_descriptor:
            module = ModuleResource.from_address(int(data.resource_descriptor))
            if cbid == ids.MODULE_UNLOAD_STARTING:
                self.on_module_unload(context, module.module_id)
            elif cbid == ids.MODULE_LOADED and module.p_cubin and module.cubin_size:
                self.on_module_load(context, module.module_id, C.string_at(module.p_cubin, module.cubin_size))

    def _launch(self, cbid, data):
        exiting = int(data.callback_site) == int(self.cupti.ApiCallbackSite.API_EXIT)
        if cbid in self._graphs:
            if exiting:
                self.on_graph_launch()
            return
        if cbid not in self._params:
            return
        key = int(data.correlation_id)
        active = getattr(self._local, "launch", None)
        # A substituted API can invoke another launch API internally.
        if active is not None and active[0] != key:
            return
        if exiting:
            if active is None:
                return
            _, launch, state = active
            address = int(data.function_return_value)
            launch.status = C.c_int.from_address(address).value if address else None
            try:
                self.on_launch_exit(launch, state)
            finally:
                try:
                    launch._restore()
                finally:
                    del self._local.launch
        else:
            launch = Launch(data, self._params[cbid])
            try:
                state = self.on_launch_enter(launch)
                if state is not None:
                    self._local.launch = (key, launch, state)
                else:
                    launch._restore()
            except BaseException:
                launch._restore()
                raise


@contextmanager
def module_loads(on_load):
    """Call on_load(context, module_id, image), serialized, with owned cubin bytes.

    Handlers should only perform CPU work. Errors surface after unsubscribe;
    a workload exception takes precedence.
    """
    lock = threading.Lock()

    def collect(context, module_id, image):
        with lock:
            on_load(context, module_id, image)

    with Callbacks(on_module_load=collect):
        yield
