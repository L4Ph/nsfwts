import { InferenceSession } from "onnxruntime-web"

const size = 224

async function loadModel(modelUrl: string): Promise<InferenceSession> {
  const session = await InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
  })
  return session
}

async function preprocessImage(
  image: HTMLImageElement,
  size: number,
): Promise<Float32Array> {
  const canvas = document.createElement("canvas")
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext("2d")
  ctx?.drawImage(image, 0, 0, size, size)
  const imageData = ctx?.getImageData(0, 0, size, size)

  const data = new Float32Array(size * size * 3)
  for (let i = 0; i < size * size; i++) {
    data[i * 3] = imageData!.data[i * 4] / 255 // R
    data[i * 3 + 1] = imageData!.data[i * 4 + 1] / 255 // G
    data[i * 3 + 2] = imageData!.data[i * 4 + 2] / 255 // B
  }
  return data
}

export async function processImageWithModel(image: HTMLImageElement, modelUrl: string) {
  const session = await loadModel(modelUrl)
  const input = await preprocessImage(image, size)
  return { session, input }
}
