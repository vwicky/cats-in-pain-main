# Training and analysis (advanced)

Commands assume the **repository root** as the current working directory. For task-to-command mapping, see also [script index](script-index.md).

| Task | Command |
|------|---------|
| YAMNet cat-prob evaluation | `python audio/audio-pre-classifier/src/evaluate_v2_data.py --help` |
| 10-class cat emotion inference | `python audio/audio-emotion-classifier/inference/04_audio_classification.py --help` |
| EfficientNet-B3 finetune | `python video/cnn-finetuning/src/train.py --config video/cnn-finetuning/config/default.yaml` |
| ViTPose extraction | `python video/easyViTPose/extraction/06_pose_extraction.py --help` |
| DeepLabCut SuperAnimal extraction | `python video/deep-lab-cut/extraction/06_pose_extraction_superanimal.py --help` |
| ST-GCN training (pose-models) | `python video/pose-models/run_stgcn_deeplabcut_train.py --config video/pose-models/config_stgcn_dlc.yaml` |
| GPT vision LLM eval (FGS) | `jupyter notebook video/gpt-fgs/cat_pain_llm_eval.ipynb` |

Subproject READMEs:

- [audio-pre-classifier](../audio/audio-pre-classifier/README.md)
- [audio-emotion-classifier](../audio/audio-emotion-classifier/README.md)
- [cnn-finetuning](../video/cnn-finetuning/README.md)
- [easyViTPose](../video/easyViTPose/README.md)
- [pose-models](../video/pose-models/) (directory)
