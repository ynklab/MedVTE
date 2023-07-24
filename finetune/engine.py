import os
import random
from pathlib import Path
from typing import Optional, Tuple, Union, Callable, Dict, List, Any

import datasets
from datasets import Image, concatenate_datasets
import evaluate
import numpy as np
import PIL
import torch
import torchvision
from transformers import (
    EvalPrediction,
    PretrainedConfig,
    Trainer,
    FlavaFeatureExtractor,
    ViltForImagesAndTextClassification,
    ViltFeatureExtractor,
    default_data_collator,
    set_seed,
    AutoConfig,
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint

from arguments import parse_arguments, validate_arguments
from data import HYPOTHESIS_KEY, IMAGE_KEY, LABEL_KEY
from flava import FlavaForImagesAndTextClassification
from logger import setup_logger
from metrics import ClassWiseAccuracy


class VisualEntailmentModelEngine:

    def __init__(self, config_json: Optional[str] = None, config_dict: Optional[dict] = None) -> None:
        # 1. Parsing command-line variables
        self.model_args, self.data_args, self.training_args = parse_arguments(config_json, config_dict)

        # 2. Setup logging
        self.logger = setup_logger(self.training_args, __name__)

        # 3. Detecting last checkpoint
        self.last_checkpoint = self._detect_last_checkpoint(self.training_args)

        # 4. Set seed before initializing model
        set_seed(self.training_args.seed)

        # 5. Prepare dataset
        self.raw_datasets, self.label_list, self.num_labels = self._load_data()

        # 6-1. Prepare model
        self.model, self.config, self.tokenizer, self.feature_extractor = self._load_model()

        self.train_dataset, self.eval_dataset, self.predict_dataset = self._process_datasets()

    def _detect_last_checkpoint(self, training_args) -> Optional[str]:
        last_checkpoint = None
        if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif (
                last_checkpoint is not None and training_args.resume_from_checkpoint is None
            ):
                self.logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        return last_checkpoint

    def _load_data(self) -> Tuple[Any, List, int]:
        validate_arguments(self.data_args, self.training_args)
        raw_datasets = self._load_and_normalize_dataset(self.data_args, self.training_args)
        raw_datasets = self._aggregate_datasets(raw_datasets)
        label_list = sorted(list(raw_datasets.values())[0].unique(LABEL_KEY))
        num_labels = len(label_list)
        return raw_datasets, label_list, num_labels

    def _load_and_normalize_dataset(self, data_args, training_args) -> datasets.DatasetDict:
        '''assume json files we will use are json lines files'''
        dict_for_dataset_dict = {}

        if training_args.do_train:
            for k, v in data_args.train_files.items():
                self.logger.info(f"load & normalize training dataset: {v}")
                dict_for_dataset_dict[f"{k}_train"] = datasets.Dataset.from_json(v)
        if training_args.do_eval:
            if data_args.do_cross_validation:
                ...
            else:
                for k, v in data_args.validation_files.items():
                    self.logger.info(f"load & normalize validation dataset: {v}")
                    dict_for_dataset_dict[f"{k}_validation"] = datasets.Dataset.from_json(v)
        if training_args.do_predict:
            for k, v in data_args.test_files.items():
                self.logger.info(f"load & normalize test dataset: {v}")
                dict_for_dataset_dict[f"{k}_test"] = datasets.Dataset.from_json(v)

        self.logger.info("Normalizing dataset keys & converting image paths")
        raw_datasets = datasets.DatasetDict(
            {
                k: self._normalize_dataset(
                    v, task_name=k.split('_')[0], image_dir=data_args.image_dirs[k.split('_')[0]]
                )
                for k, v in dict_for_dataset_dict.items()
            }
        )
        return raw_datasets

    def _normalize_dataset(
        self, dataset: datasets.Dataset, task_name: str, image_dir: str
    ) -> datasets.Dataset:
        # Rename columns
        hypothesis_key = self._get_hypothesis_key(task_name)
        if HYPOTHESIS_KEY not in dataset.features:  # TODO: use correct way to check keys
            dataset = dataset.rename_column(hypothesis_key, HYPOTHESIS_KEY)
        label_key = self._get_label_key(task_name)
        if LABEL_KEY not in dataset.features:  # TODO: use correct way to check keys
            dataset = dataset.rename_column(label_key, LABEL_KEY)

        # Set image path for IMAGE_KEY
        dataset = dataset.map(
            lambda example: self._set_image_path(
                example, task_name=task_name, image_dir=image_dir
            )
        )
        # Set PIL.Image object for IMAGE_KEY
        dataset = dataset.cast_column(IMAGE_KEY, Image())

        # Remove extra columns
        dataset = dataset.remove_columns(
            [
                col
                for col in dataset.features.keys()
                if col not in (HYPOTHESIS_KEY, LABEL_KEY, IMAGE_KEY)
            ]
        )
        return dataset

    @staticmethod
    def _get_hypothesis_key(task_name: str) -> str:
        if task_name == "snli-ve":
            return "sentence2"
        elif task_name in ['medvte-strict', 'medvte-loose']:
            return 'hypothesis'
        else:
            raise NotImplementedError('add new elif branch if you would like to add new task')

    @staticmethod
    def _get_label_key(task_name: str) -> str:
        if task_name == "snli-ve":
            return "gold_label"
        elif task_name == 'medvte-loose':
            return 'loose_label'
        elif task_name == 'medvte-strict':
            return 'strict_label'
        else:
            raise NotImplementedError('add new elif branch if you add new task')

    @staticmethod
    def _set_image_path(example: Dict, task_name: str, image_dir: str) -> Dict:
        if task_name == "snli-ve":
            example[IMAGE_KEY] = str(Path(image_dir) / example["captionID"].split("#")[0])
        elif task_name in ['medvte-strict', 'medvte-loose']:
            example[IMAGE_KEY] = str(Path(image_dir) / example['fig_basename'])
        else:
            raise NotImplementedError
        return example

    @staticmethod
    def _aggregate_datasets(data: datasets.DatasetDict) -> datasets.DatasetDict:
        train_datasets = [v for k, v in data.items() if k.endswith('_train')]
        validation_datasets = [v for k, v in data.items() if k.endswith('_validation')]
        test_ds = concatenate_datasets([v for k, v in data.items() if k.endswith('_test')])

        if train_datasets and validation_datasets:
            train_ds = concatenate_datasets(train_datasets)
            validation_ds = concatenate_datasets(validation_datasets)
            return datasets.DatasetDict(
                {
                    'train': train_ds,
                    'validation': validation_ds,
                    'test': test_ds
                }
            )
        elif train_datasets:
            train_ds = concatenate_datasets(train_datasets)
            return datasets.DatasetDict(
                {
                    'train': train_ds,
                    'test': test_ds
                }
            )
        elif validation_datasets:
            validation_ds = concatenate_datasets(validation_datasets)
            return datasets.DatasetDict(
                {
                    'validation': validation_ds,
                    'test': test_ds
                }
            )
        else:
            return datasets.DatasetDict(
                {
                    'test': test_ds
                }
            )

    def _load_model(self) -> None:
        config = AutoConfig.from_pretrained(
            self.model_args.config_name
                if self.model_args.config_name
                else self.model_args.model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task='+'.join(self.data_args.task_names),
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        self.logger.info(config)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name
            if self.model_args.tokenizer_name
            else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        self.logger.info(tokenizer)
        if 'vilt' in self.model_args.model_name_or_path:
            feature_extractor = ViltFeatureExtractor.from_pretrained(
                "dandelin/vilt-b32-mlm"
            )  # noqa
            # _ = ViltProcessor.from_pretrained(self.model_args.model_name_or_path)  # noqa
            model = ViltForImagesAndTextClassification.from_pretrained(
                self.model_args.model_name_or_path, num_images=1, num_labels=self.num_labels
            )
        elif 'flava' in self.model_args.model_name_or_path:
            # _ = FlavaProcessor.from_pretrained(self.model_args.model_name_or_path)  # noqa
            feature_extractor = FlavaFeatureExtractor('facebook/flava-full')
            model = FlavaForImagesAndTextClassification.from_pretrained(
                self.model_args.model_name_or_path, num_labels=self.num_labels
            )
        else:
            raise NotImplementedError
        self.logger.info(feature_extractor)
        self.logger.info(model)

        return model, config, tokenizer, feature_extractor

    def _process_datasets(self) -> Tuple[Any, Any, Any]:
        # 6-3. Comparison of data_args.max_length and tokenizer.max_length
        if self.data_args.max_seq_length > self.tokenizer.model_max_length:
            self.logger.warning(
                f"The max_seq_length passed ({self.data_args.max_seq_length}) "
                + "is larger than the maximum length for the"
                + f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
            )
        max_seq_length = min(self.data_args.max_seq_length, self.tokenizer.model_max_length)

        # 6-4. Label order
        # Some models have set the order of the labels to use, so let's make sure we do use it.
        label_to_id = self._normalize_label_order(self.label_list, self.num_labels, self.model, self.data_args)
        self.label_to_id = label_to_id

        if label_to_id is not None:
            self.model.config.label2id = label_to_id
            self.model.config.id2label = {id: label for label, id in self.config.label2id.items()}
        elif self.data_args.task_names is not None:
            self.model.config.label2id = {label: i for i, label in enumerate(self.label_list)}
            self.model.config.id2label = {id: label for label, id in self.config.label2id.items()}

        if self.training_args.do_train:
            self.logger.debug(self.raw_datasets["train"][0])

        # 6-5. Load images and convert labels to ids
        with self.training_args.main_process_first(desc="dataset map pre-processing"):
            # 6-2. Padding strategy
            if self.data_args.pad_to_max_length:
                padding = "max_length"
            else:
                # We will pad later, dynamically at batch creation, to the max sequence length in each batch
                padding = False

            tokenized_datasets = self.raw_datasets.map(
                self._preprocess_data_and_label(padding, max_seq_length),
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        tokenized_datasets = tokenized_datasets.remove_columns([HYPOTHESIS_KEY])

        if self.training_args.do_train:
            print("=" * 5 + "SAMPLE AFTER APPLYING TOKENIZATION" + "=" * 5)
            print(tokenized_datasets["train"][0])

        # convert image to image feature
        tokenized_datasets.set_transform(
            self._extract_image_feature, columns=[IMAGE_KEY], output_all_columns=True
        )

        if self.training_args.do_train:
            print("=" * 5 + "SAMPLE AFTER SETTING TRANSFORM" + "=" * 5)
            print(tokenized_datasets["train"][0])

        # 6-6. Data collator
        # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
        # we already did the padding.

        if self.data_args.pad_to_max_length:
            if "vilt" in self.model_args.model_name_or_path:
                data_collator = default_data_collator
                # data_collator = None
            else:
                data_collator = default_data_collator
        else:
            data_collator = None
        self.data_collator = data_collator

        # 6-7. Control dataset size
        # raw_train_dataset, raw_eval_dataset, raw_predict_dataset = self._get_sliced_datasets(
        #     self.raw_datasets, self.data_args, self.training_args
        # )
        train_dataset, eval_dataset, predict_dataset = self._get_sliced_datasets(
            tokenized_datasets, self.data_args, self.training_args
        )

        # 6-8. Samples
        # Log a few random samples from the training set:
        if self.training_args.do_train:
            for index in random.sample(range(len(train_dataset)), 3):
                self.logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        return train_dataset, eval_dataset, predict_dataset

    def _preprocess_data_and_label(self, padding, max_seq_length) -> Callable:
        def func(examples: List[Dict]) -> List[Dict]:
            # Process batch texts
            texts = examples[HYPOTHESIS_KEY]
            result = self.tokenizer(
                text=texts,
                padding=padding,
                max_length=max_seq_length,
                truncation=True,
            )

            # Map labels to IDs
            if self.label_to_id is not None and LABEL_KEY in examples:
                result[LABEL_KEY] = [
                    (self.label_to_id[label] if label != -1 else -1)
                    for label in examples[LABEL_KEY]
                ]
            return result
        return func

    def _extract_image_feature(
        self, batch: Dict[str, List]
    ) -> Dict[str, Union[List[int], torch.Tensor]]:
        images: List[PIL.Image] = batch[IMAGE_KEY]

        if 'vilt' in self.model_args.model_name_or_path:
            landscape_images = list(map(self._rotate_and_resize, images))
            with torch.autocast("cuda"):
                image_feature = self.feature_extractor(landscape_images, return_tensors="pt")
            # unsqueeze because ViltForImagesAndTextClassification model requires "num_image" axis
            image_feature = {k: v.unsqueeze(0) for k, v in image_feature.items()}
        elif 'flava' in self.model_args.model_name_or_path:
            with torch.autocast("cuda"):
                image_feature = self.feature_extractor(images, return_tensors='pt')
        else:
            raise NotImplementedError(f'extraction of image feature is not implemented')
        return image_feature

    @staticmethod
    def _rotate_and_resize(image: PIL.Image) -> PIL.Image:
        # 1. Rotate 90 deg portrait image -> landscape image
        # 2. Resize with height = 384px
        # 3. Pad and crop with width = 640px
        TARGET_HEIGHT = 384
        TARGET_WIDTH = 640
        if image.height > image.width:
            image = torchvision.transforms.functional.rotate(
                image, angle=90, expand=True
            )
        scale = TARGET_HEIGHT / image.height
        image = image.resize((int(image.width * scale), TARGET_HEIGHT))
        image = image.crop((0, 0, TARGET_WIDTH, TARGET_HEIGHT))
        return image

    def _normalize_label_order(
        self, label_list, num_labels, model, data_args
    ) -> Dict[Union[str, int], int]:
        if data_args.task_names is not None:
            if model.config.label2id == PretrainedConfig(num_labels=num_labels).label2id:
                label_to_id = {v: i for i, v in enumerate(label_list)}
            else:
                # Some have all caps in their config, some don't.
                label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
                if sorted(label_name_to_id.keys()) == sorted(label_list):
                    label_to_id = {v: label_name_to_id[v] for v in label_list}
                else:
                    self.logger.warning(
                        "Your model seems to have been trained with labels, but they don't match the dataset: ",
                        f"model labels: {list(sorted(label_name_to_id.keys()))}, "
                        + "dataset labels: {list(sorted(label_list))}."
                        "\nIgnoring the model labels as a result.",
                    )
                    label_to_id = {v: i for i, v in enumerate(label_list)}
        else:
            label_to_id = {v: i for i, v in enumerate(label_list)}
        return label_to_id

    @staticmethod
    def _get_sliced_datasets(
        raw_datasets: datasets.DatasetDict, data_args, training_args
    ) -> Tuple[
        datasets.Dataset,
        datasets.Dataset,
        datasets.Dataset,
    ]:
        train_dataset, eval_dataset, predict_dataset = (None, None, None)

        if training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

        if training_args.do_eval and not data_args.do_cross_validation:
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))

        if (
            training_args.do_predict
            or data_args.task_names is not None
            or data_args.test_files is not None
        ):
            if "test" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                max_predict_samples = min(
                    len(predict_dataset), data_args.max_predict_samples
                )
                predict_dataset = predict_dataset.select(range(max_predict_samples))

        return train_dataset, eval_dataset, predict_dataset

    def _compute_metrics(self) -> Callable:
        # 6-9. Metric
        metric_names = ["accuracy", "recall", "precision", "f1"]
        metric_maps = {name: evaluate.load(name) for name in metric_names}

        # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
        # predictions and label_ids field) and has to return a dictionary string to float.
        def func(p: EvalPrediction) -> Dict[str, float]:
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            if self.data_args.task_names is not None:
                results = {}
                for name, metric in metric_maps.items():
                    if name == "accuracy":
                        # Class-wise results
                        split_result: Dict[str, List[float]] = ClassWiseAccuracy().compute(
                            predictions=preds,
                            references=p.label_ids,
                            labels=list(self.label_to_id.values()),
                        )
                        for label_name, label_ix in self.label_to_id.items():
                            results[f"{name}_{label_name}"] = list(split_result.values())[
                                0
                            ][label_ix]
                        # Aggregated results
                        agg_result: Dict[str, float] = metric.compute(
                            predictions=preds, references=p.label_ids
                        )
                        results[name] = list(agg_result.values())[0]
                    else:
                        # Class-wise results
                        split_result: Dict[str, List[float]] = metric.compute(
                            predictions=preds,
                            references=p.label_ids,
                            average=None,
                            labels=list(self.label_to_id.values()),
                        )
                        for label_name, label_ix in self.label_to_id.items():
                            results[f"{name}_{label_name}"] = list(split_result.values())[
                                0
                            ][label_ix]
                        # Aggregated results
                        micro_result: Dict[str, float] = metric.compute(
                            predictions=preds,
                            references=p.label_ids,
                            average="micro",
                            labels=list(self.label_to_id.values()),
                        )
                        macro_result: Dict[str, float] = metric.compute(
                            predictions=preds,
                            references=p.label_ids,
                            average="macro",
                            labels=list(self.label_to_id.values()),
                        )
                        results[f"{name}_micro_avg"] = list(micro_result.values())[0]
                        results[f"{name}_macro_avg"] = list(macro_result.values())[0]
                return results
            else:
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
        return func

    def train(self, train_dataset: Optional[datasets.Dataset] = None, eval_dataset: Optional[datasets.Dataset] = None, output_dir: Optional[str] = None) -> None:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if output_dir is not None:
            old_output_dir = self.training_args.output_dir
            self.training_args.output_dir = output_dir

        # 7. Running
        # Initialize our Trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset if self.training_args.do_train else None,
            eval_dataset=eval_dataset if self.training_args.do_eval else None,
            compute_metrics=self._compute_metrics(),
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        # Training
        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            max_train_samples = (
                self.data_args.max_train_samples
                if self.data_args.max_train_samples is not None
                else len(self.train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.save_model()  # Saves the tokenizer too for easy upload

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if self.training_args.do_eval:
            self.logger.info("*** Evaluate ***")

            # Loop to handle MNLI double evaluation (matched, mis-matched)
            task = '+'.join(self.data_args.task_names)
            eval_ds = eval_dataset

            metrics = trainer.evaluate(eval_dataset=eval_ds)

            max_eval_samples = (
                self.data_args.max_eval_samples
                if self.data_args.max_eval_samples is not None
                else len(eval_ds)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_ds))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        # Prediction
        if self.training_args.do_predict:
            self._predict(trainer)

        kwargs = {
            "finetuned_from": self.model_args.model_name_or_path,
            "tasks": "text-classification",
        }
        if self.data_args.task_names is not None:
            kwargs["language"] = "en"
            kwargs["dataset_tags"] = "vte"
            kwargs["dataset_args"] = '+'.join(self.data_args.task_names)
            kwargs["dataset"] = "VTE"

        if output_dir is not None:
            self.training_args.output_dir = old_output_dir

        if self.training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)

    def predict(self) -> None:
        # Initialize our Trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=None,
            compute_metrics=self._compute_metrics(),
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        self._predict(trainer)

    def _predict(self, trainer) -> None:
        self.logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        task = '+'.join(self.data_args.task_names)
        predict_dataset = self.predict_dataset

        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        # predict_dataset = predict_dataset.remove_columns(LABEL_KEY)
        predict_tuple = trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        )
        predictions = predict_tuple.predictions

        metrics = self._compute_metrics()(predict_tuple)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        predictions = np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(
            self.training_args.output_dir, f"predict_results_{task}.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                self.logger.info(f"***** Predict results {task} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = self.label_list[item]
                    writer.write(f"{index}\t{item}\n")
