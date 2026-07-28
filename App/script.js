ort.env.wasm.wasmPaths = "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.17.0/";
ort.env.wasm.numThreads = 1;

let session = null;

async function loadModel() {
  try {
    session = await ort.InferenceSession.create("./model.onnx");
    console.log("Model loaded successfully");
    console.log("Input names:", session.inputNames);
    console.log("Output names:", session.outputNames);
  } catch (err) {
    console.error("Model failed to load:", err);
    document.getElementById("result").innerText = "Error loading model: " + err.message;
  }
}

async function predict() {
  if (!session) {
    document.getElementById("result").innerText = "Model not loaded yet.";
    return;
  }
  const f1 = parseFloat(document.getElementById("feature1").value);
  const f2 = parseFloat(document.getElementById("feature2").value);
  const f3 = parseFloat(document.getElementById("feature3").value);

  const inputTensor = new ort.Tensor("float32", Float32Array.from([f1, f2, f3]), [1, 3]);
  const feeds = { [session.inputNames[0]]: inputTensor };
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]];
  document.getElementById("result").innerText = "Prediction: " + output.data[0];
}

document.getElementById("predictBtn").addEventListener("click", predict);
loadModel();