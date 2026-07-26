import skops.io as sio
from skl2onnx import to_onnx
import numpy as np

pipe = sio.load("./Model/model_pipeline.skops", trusted=True)
sample_input = np.zeros((1, 3), dtype=np.float32)
onnx_model = to_onnx(pipe, sample_input)

with open("./App/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())