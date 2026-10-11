# Shared-Representation Digit Model

A small multimodal project inspired by MiniMax H3. One model maps digit images and digit words into a single shared vector space, then recognizes and generates across both.

## What it does

The model has two input paths, one for images and one for words, which feed into the same shared core. From that shared representation it can:

- **Recognize images:** classify a handwritten digit (0 to 9) from MNIST.
- **Recognize words:** map the words "zero" to "nine" to their digit.
- **Generate images from words:** turn a word into a 28x28 digit image, then check whether the image path recognizes it as the same digit.

Training uses five losses at once: image classification, word classification, image reconstruction, word-to-image generation, and an alignment loss that pulls the image and word vectors for the same digit together.

## How it differs from MiniMax H3

MiniMax H3 is a large omni-modal model that handles text, image, and audio in one system, trained on very large datasets with substantial compute. This project is a small proof of the same core idea: one shared representation used by more than one modality. It uses two modalities (images and words), ten classes, a small convolutional encoder, and a short training run on a laptop CPU. It does not attempt to match H3's scale, quality, or range of outputs.

## Results

Trained for 5 epochs with the Adam optimizer on the MNIST training set, evaluated on the MNIST test set:

| Metric | Score |
|---|---|
| Image recognition accuracy (test set) | 0.988 |
| Word recognition accuracy | 1.000 |
| Word → image → recognized accuracy | 1.000 |

Word recognition is near-perfect because the word task is a lookup over ten fixed inputs, so it is easy by design. The word → image → recognized score is the more meaningful result: it shows that images generated from words are recognized by the image path as the correct digit, which means the two paths share a representation.

![Generated digits from words](images/generated_from_words.png)
## Limitations

- Digits only: ten classes, no open vocabulary.
- Two modalities: images and words. Audio and video are not included.
- Generated images are blurry, as expected from a small decoder trained for a few epochs.
- Short training run (5 epochs) on a CPU.
- The word path is a lookup table, so it does not learn language.

## Next steps

- Add audio using the Speech Commands dataset and spectrograms, so the same shared core handles a third modality.
- Expand the vocabulary beyond ten digits, using a larger text dataset.
- Replace the small convolutional decoder with a stronger generator and train for longer.
- Compare the shared model against separate single-modality models to measure whether sharing helps.

   ## How to run

```
   pip install torch torchvision matplotlib
   python main.py
```

The first run downloads MNIST into `data/`. Outputs are saved to `outputs/`: the trained model (`model.pt`) and the generated image grid.