# nsfwts

Alternative to [infinitered/nsfwjs](https://github.com/infinitered/nsfwjs)

# Model List
- [deepghs/imgutils-models](https://huggingface.co/deepghs/imgutils-models/blob/main/nsfw/nsfwjs.onnx)

# Image Classification with ONNX Runtime Web

This project demonstrates how to classify images using a pre-trained model with ONNX Runtime Web.

## Usage

Here is an example of how to use the provided `load` function to classify an image:

```typescript
import { load } from './path-to-your-script';

const image = document.getElementById('your-image-id') as HTMLImageElement;
const modelUrl = '@l4ph/nsfwts';

load(image, modelUrl).then(results => {
  console.log('Classification Results:', results);
}).catch(error => {
  console.error('Error during classification:', error);
});
```

## API

### `load(image: HTMLImageElement, modelUrl: string): Promise<Record<string, number>>`

Classifies an input image using the specified ONNX model.

#### Parameters

- `image`: An `HTMLImageElement` representing the image to be classified.
- `modelUrl`: A string representing the URL to the ONNX model.

#### Returns

A `Promise` that resolves to a `Record<string, number>`, where the keys are the classification labels and the values are the corresponding probabilities.

## Example

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Image Classification Example</title>
</head>
<body>
  <img id="your-image-id" src="path-to-your-image.jpg" alt="Image for Classification">
  <script type="module">
    import { load } from '@l4ph/nsfwts';

    const image = document.getElementById('your-image-id');
    const modelUrl = 'path-to-your-model.onnx';

    load(image, modelUrl).then(results => {
      console.log('Classification Results:', results);
    }).catch(error => {
      console.error('Error during classification:', error);
    });
  </script>
</body>
</html>
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
