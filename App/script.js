let session = null;

async function loadModel() {
  session = await ort.InferenceSession.create("./model.onnx");
}

async function predict() {
  const f1 = parseFloat(document.getElementById("feature1").value);
  const f2 = parseFloat(document.getElementById("feature2").value);
  const f3 = parseFloat(document.getElementById("feature3").value);

  const inputTensor = new ort.Tensor("float32", Float32Array.from([f1, f2, f3]), [1, 3]);
  const feeds = { float_input: inputTensor };
  const results = await session.run(feeds);
  const output = results[Object.keys(results)[0]];
  document.getElementById("result").innerText = "Prediction: " + output.data[0];
}

document.getElementById("predictBtn").addEventListener("click", predict);
loadModel();