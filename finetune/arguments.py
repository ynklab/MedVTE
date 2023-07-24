from dataclasses import dataclass, field
import os
import sys
from typing import Optional, Tuple, Dict, List

from transformers import HfArgumentParser, TrainingArguments

TASK_NAMES = [
    "snli-ve",
    'medvte-loose',
    'medvte-strict'
]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    task_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of the names of the task to train on: " + ", ".join(TASK_NAMES)},
    )

    train_files: Optional[Dict[str, str]] = field(
        default=None,
        metadata={"help": "A dict of key: dataset_name and value: json dataset file containing the training data."},
    )
    validation_files: Optional[Dict[str, str]] = field(
        default=None,
        metadata={"help": "A dict of key: dataset_name and value: json dataset file containing the validation data."},
    )
    test_files: Optional[Dict[str, str]] = field(
        default=None,
        metadata={"help": "A dict of key: dataset_name and value: json dataset file containing the test data."},
    )
    image_dirs: Optional[Dict[str, str]] = field(
        default=None, metadata={"help": "A dict of directories containing image data."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    do_cross_validation: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to do cross-validation"
                "if true, will do k-fold cross validation using training data"
            )
        },
    )

    def __post_init__(self):
        # check if task name is in predefined TASK_NAMES
        if self.task_names is not None:
            self.task_names = [task.lower() for task in self.task_names]
            for task in self.task_names:
                if task not in TASK_NAMES:
                    raise ValueError(
                        "Unknown task, you should pick one in " + ",".join(TASK_NAMES)
                    )

        # check all the files have correct extensions
        if self.train_files is None:
            return
        for k, v in self.train_files.items():
            train_extension = v.split('.')[-1]
            assert train_extension in [
                "json",
                "jsonl",
            ], "`train_file` should be a json or a jsonl file."

            if not self.do_cross_validation and self.validation_files is not None:
                validation_file = self.validation_files.get(k, '')
                # assert (
                #     validation_file
                # ), "`validation_files` must be defined when corersponding train_files exist."
                if validation_file:
                    validation_extension = validation_file.split(".")[-1]
                    assert (
                        validation_extension == train_extension
                    ), "`validation_file` should have the same extension (json or jsonl) as `train_file`."

            if self.test_files is not None:
                test_extension = self.test_files.get(k, '').split(".")[-1]
                if not test_extension:
                    continue
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (json or jsonl) as `train_file`."


def parse_arguments(config_json: Optional[str], config_dict: Optional[dict] = None) -> Tuple[
    ModelArguments, DataTrainingArguments, TrainingArguments
]:
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if config_dict is not None:
        model_args, data_args, training_args = parser.parse_dict(
            args=config_dict
        )
    elif config_json is not None:
        # for debugging use case
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=config_json
        )
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        raise ValueError("Select a config file for finetuning")

    return model_args, data_args, training_args


def validate_arguments(data_args, training_args) -> None:
    if training_args.do_predict and data_args.test_files is None:
        raise ValueError("Need a test file for `do_predict`.")
