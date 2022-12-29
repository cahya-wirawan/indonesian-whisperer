# Indonesian Whisperer
Experiment with OpenAI Whisper on Indonesian Languages

## OpenAI Whisper

## Development

### Dependencies Installation

### Fine Tuning
We fine-tuned the original OpenAI Whisper with several Indonesian datasets.

| Model                                                                       | WER on CV11 |
|-----------------------------------------------------------------------------|-------------|
| [Indonesian Whisper Tiny](https://huggingface.co/cahya/whisper-tiny-id)     | 18.28       |
| [Indonesian Whisper Small](https://huggingface.co/cahya/whisper-small-id)   | 6.06        |
| [Indonesian Whisper Medium](https://huggingface.co/cahya/whisper-medium-id) | 3.83        |
| [Indonesian Whisper Large](https://huggingface.co/cahya/whisper-large-id)   | 6.25*       |

The original OpenAI Whisper Medium model has WER of 12.x, but we got 3.83 after fine-tuning it with Indonesian datasets.

*The WER of Indonesian Whisper Large is worst than the Medium and Small model because we fine-tuned it with fewer 
epochs than the other models.

### Inference Pipeline

## Authors

Following are the authors of this work (listed alphabetically):

