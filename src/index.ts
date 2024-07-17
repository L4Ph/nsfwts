import { Tensor } from "onnxruntime-web";
import { processImageWithModel } from "./setup";
import { NSFW_CLASSES } from "./nsfw_classes"

export type predictionType = {
  className: (typeof NSFW_CLASSES)[keyof typeof NSFW_CLASSES];
  probability: number;
};

const size = 224;

export async function load(image: HTMLImageElement, modelUrl: string): Promise<Array<predictionType>> {
  const { session, input } = await processImageWithModel(image, modelUrl);
  const tensor = new Tensor("float32", input, [1, size, size, 3]);
  const feeds: Record<string, Tensor> = { input_1: tensor };
  const output = await session.run(feeds);
  const outputData = output.dense_3.data as Float32Array;

  const predictions: Array<predictionType> = [];
  outputData.forEach((probability, index) => {
    predictions.push({
      className: NSFW_CLASSES[index],
      probability
    });
  });
  return predictions;
}
