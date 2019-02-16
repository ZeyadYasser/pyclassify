# pyclassify
Simple command line tool for transfer learning

## Installation
```bash
pip install git+https://github.com/ZeyadYasser/pyclassify
```
---

## Training
```bash
PyClassify for transfer learning

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint-dir CHECKPOINT_DIR
                        Directory of latest checkpoint (default: none)
  --save-dir SAVE_DIR   Directory to save model checkpoints
  --data-dir DATA_DIR   Directory to data:
                                root/train/dog/xxx.png
                                root/train/cat/yyy.png

                                root/val/dog/xxx.png
                                root/val/cat/yyy.png
  --epochs EPOCHS       # of epochs to run (default: 50)
  --batch-size BATCH_SIZE
                        Batch size (default: 32)
  --num-workers NUM_WORKERS
                        # of CPU workers that prefetch data (default: 4)
  --device {cuda,cpu}   cuda or cpu (default: cpu)
  --backend-model BACKEND_MODEL
                        Backend model to use (default: squeeze_net)
  --model-name MODEL_NAME
                        Name for your new model (default: image_classifier)
  --lr LR, --learning-rate LR
                        Model learning rate (default: 0.001)
  --momentum MOMENTUM   Model momentum (default: 0.9)
  --weight-decay WEIGHT_DECAY
                        Model weight decay (default: 0.0002)
```

Example :
```bash
pyclassify_train
	--checkpoint-dir=path/to/model_checkpoint
	--save-dir=path/to/model_checkpoint
	--data-dir=path/to/data
	--epochs=30
	--batch-size=64
	--num-workers=4
	--device=cuda
	--backend-model=squeeze_net
	--model-name=cat_dog_classifier
	--lr=0.001
	--momentum=0.9
	--weight-decay=0.0002
```
---

## Classification
```bash
PyClassify for transfer learning

positional arguments:
  checkpoint_dir       Directory of model checkpoint
  image_dir            Path to image

optional arguments:
  -h, --help           show this help message and exit
  --device {cuda,cpu}  cuda or cpu (default: cpu)
```

Example :
```bash
pyclassify_run
	path/to/model_checkpoint
	path/to/img
```