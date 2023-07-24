## Create environment
```
$ poetry install
```

## Finetune
- place datasets (jsonl, images) to data directory as below
```
├── README.md
├── data
│   ├── flickr30k_images/ <- flickr images for SNLI-VE
│   ├── medvte-test.jsonl
│   ├── medvte-train.jsonl
│   ├── medvte.jsonl
│   ├── medvte_figures/ <- MedICaT figures for MedVTE
│   ├── snli_ve_dev.jsonl
│   ├── snli_ve_test.jsonl
│   └── snli_ve_train.jsonl
├── finetune
│   ├── README.md
│   ├── arguments.py
│   ├── config
│   ├── data.py
│   ├── engine.py
│   ├── flava.py
│   ├── logger.py
│   ├── metrics.py
│   └── run.py
├── poetry.lock
├── pyproject.toml
└── static
    ├── medvte_overview_github.png
    └── medvte_two_labels.png
```
- select a config file as an argument of python command
```
$ poetry run python finetune/run.py finetune/config/vilt/snlive-medvte-loose.json
```
