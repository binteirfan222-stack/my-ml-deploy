import skops.io as sio
from skl2onnx import to_onnx
import numpy as np
import os

os.makedirs("./App", exist_ok=True)

untrusted_types = sio.get_untrusted_types(file="./Model/model_pipeline.skops")
pipe = sio.load("./Model/model_pipeline.skops", trusted=untrusted_types)

sample_input = np.zeros((1, 3), dtype=np.float32)
onnx_model = to_onnx(pipe, sample_input)

with open("./App/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())