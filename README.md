# Whisper Finetune 1 Notebook

In this experiment, Whisper (base) is finetuned on VinBigData 100h dataset, but with special pre-processing:
- Remove sentence with `<unk>` token (The data is clean and good compare to other open source Vietnamese data, but the transcript is the output of a larger model from Vinbigdata - Kaldi I think. I don't know if it is later verified by human but a few of them still contain `<unk>` token)
- Punctuation and Capitalization restoration by [dragonSwing/xlm-roberta-capu](https://huggingface.co/dragonSwing/xlm-roberta-capu)
- Spoken to written transcript [nguyenvulebinh/spoken-norm](https://github.com/nguyenvulebinh/spoken-norm)

As state in the [paper](https://arxiv.org/pdf/2212.04356.pdf):
> Recent research has shown that training on datasets of mixed human and machine-generated data can significantly impair the performance of translation systems (Ghorbani et al., 2021). In order to avoid learning “transcript-ese”, we developed many heuristics to detect and remove machine-generated transcripts from the training datase

Whisper output is already in written form, and we would want to keep this ability by doing the last 2 preprocessing step. **However, the result is not perfect**.

## Usage
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_trained = WhisperForConditionalGeneration.from_pretrained('hkab/whisper-base-vietnamese-finetuned')
processor = WhisperProcessor.from_pretrained("hkab/whisper-base-vietnamese-finetuned")

forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")

input_speech, rate = librosa.load('/path/to/audio.wav', sr=16000)
input_features = processor(input_speech, sampling_rate=rate, return_tensors="pt").input_features

predicted_ids = model_trained.generate(input_features, forced_decoder_ids=forced_decoder_ids)

print(f'Prediction: {processor.batch_decode(predicted_ids, skip_special_tokens=True)}')
```


## Installation

In case you don't know how to use [nguyenvulebinh/spoken-norm](https://github.com/nguyenvulebinh/spoken-norm). Use this [Docker](https://hub.docker.com/r/huggingface/transformers-pytorch-gpu/tags?page=1&name=4.17) and this [Inference code](https://huggingface.co/spaces/nguyenvulebinh/spoken-norm-taggen/tree/main)
## Sample results

Origin pred | Finetune pred | References |
|--|--|--|
|Văn hóa và bộ lạc cổ sưa đã bắt đầu dữ những công bật này để dễ lấy sữa tóc, thịt và da.|Văn hóa và bộ lạc cổ xưa đã bắt đầu giữ những con vật này để dễ lấy sữa tóc thực và gia.|Văn hóa và bộ lạc cổ xưa đã bắt đầu giữ những con vật này để dễ lấy sữa, tóc, thịt, và da.|
|Nói My Spring Book, Tăng này đã giúp đời tiếng các thúc chối thuân 5 chẳng liền.|Đối với Sờ riêng Búc, trận này đã giúp đỡ tiếng các tốc chúi thua 5 chặng liền.|Đối với Springboks, trận này đã giúp đội tuyển kết thúc chuỗi thua 5 trận liền.|
|Nó cũng tấn công mọi thứ trong nước, ngày cả khủng lòng khẩn lò như tí vết cũng không phải là đối thủ với nó.|Nó cũng tấn công mọi thứ trong nước, ngay cả khủng long khổng lồ như Tricic cũng không phải là đối thủ bên nó.|Nó cũng tấn công mọi thứ trong nước; ngay cả khủng long khổng lồ như T. rex cũng không phải là đối thủ với nó.|
|Xong người có mặt quá lớn nên việc mọi người tới được nơi tổ chức lễ tan tại Quảng Trường Thánh Bí Tơ là không thể.|Số người có mặt quá lớn nên việc mọi người tới được nơi tổ chức lễ tang tại Quảng Trường Thánh B tơ là không thể.|Số người có mặt quá lớn nên việc mọi người tới được nơi tổ chức lễ tang tại Quảng trường Thánh Peter là không thể.|
|Cho ó là lục địa tương nói nhỏ nhưng có nhiều quốc gia đục lập. Thông thường, việc đi qua nhiều quốc gia đồng nghỉa với việc phải xinh thị thức và sách, họ chứ nhiều lần.|Cho là lục địa tương đối nhỏ nhưng có nhiều quốc gia độc lập, thông thường việc đi qua nhiều quốc gia đồng nghĩa với việc phải xin thị thực và xét hoàn chiếu nhiều lần.|Châu Âu là lục địa tương đối nhỏ nhưng có nhiều quốc gia độc lập. Thông thường, việc đi qua nhiều quốc gia đồng nghĩa với việc phải xin thị thực và xét hộ chiếu nhiều lần.|


## Performance on Fleurs (800 test audio)

|Metric|openai/whisper-base|whisper-base finetuned|
|--|--|--|
|Un-normalized text WER|51%|35%|

Training parameters (full in `train.py`):
```
batch_size = 16
num_epochs = 10
learning_rate=5e-4
warmup_steps=2000,
```