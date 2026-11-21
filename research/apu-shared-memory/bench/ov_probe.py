import openvino as ov

core = ov.Core()
print("devices:", core.available_devices)
for d in core.available_devices:
    try:
        print("==", d, "| full name:", core.get_property(d, "FULL_DEVICE_NAME"))
    except Exception as e:
        print("==", d, "(no name)", e)
try:
    print("NPU performance mode:", core.get_property("NPU", "SUPPORTED_PERF_COUNT"),
          "tiles:", core.get_property("NPU", "SUPPORTED_TILE_NUM"))
except Exception as e:
    print("NPU props:", e)
