import { Tensor } from "onnxruntime-web"
import { processImageWithModel } from "./setup"

const labels = ["drawings", "hentai", "neutral", "porn", "sexy"]
const size = 224

export async function load(image: HTMLImageElement, modelUrl: string): Promise<Record<string, number>> {
  const { session, input } = await processImageWithModel(image, modelUrl)
  const tensor = new Tensor("float32", input, [1, size, size, 3])
  const feeds: Record<string, Tensor> = { input_1: tensor }
  const output = await session.run(feeds)
  const outputData = output.dense_3.data as Float32Array
  const results: Record<string, number> = {}
  labels.forEach((label, index) => {
    results[label] = outputData[index]
  })
  return results
}
