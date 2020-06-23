import os
import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf


def write_data_configs(output_base_path):
    all_configs = {
        "mnli": {
            "task": "mnli",
            "paths": {
                "train": "mnli/train.jsonl",
                "val": "mnli/val.jsonl",
                "test": "mnli/test.jsonl",
            },
            "name": "mnli",
        },
        "ccg": {
            "task": "ccg",
            "paths": {
                "train": "ccg/ccg.train",
                "val": "ccg/ccg.dev",
                "test": "ccg/ccg.test",
                "tags_to_id": "ccg/tags_to_id.json",
            },
            "name": "ccg",
        },
        "squadv1": {
            "task": "squadv1",
            "paths": {
                "train": "squadv1/train-v1.1.json",
                "val": "squadv1/dev-v1.1.json",
                "test": "squadv1/dev-v1.1.json",
            },
            "name": "squadv1",
        },
        "cosmosqa": {
            "task": "cosmosqa",
            "paths": {
                "train": "cosmosqa/train.csv",
                "val": "cosmosqa/valid.csv",
                "test": "cosmosqa/test_no_label.csv",
            },
            "name": "cosmosqa",
        },
        "rte": {
            "task": "rte",
            "paths": {
                "train": "rte/train.jsonl",
                "val": "rte/val.jsonl",
                "test": "rte/test.jsonl",
            },
            "name": "rte",
        },
        "cola": {
            "task": "cola",
            "paths": {
                "train": "cola/train.jsonl",
                "val": "cola/val.jsonl",
                "test": "cola/test.jsonl",
            },
            "name": "cola",
        },
        "boolq": {
            "task": "boolq",
            "paths": {
                "train": "boolq/train.jsonl",
                "val": "boolq/val.jsonl",
                "test": "boolq/test.jsonl",
            },
            "name": "boolq",
        },
        "wic": {
            "task": "wic",
            "paths": {
                "train": "wic/train.jsonl",
                "val": "wic/val.jsonl",
                "test": "wic/test.jsonl",
            },
            "name": "wic",
        },
    }
    for task_name, config in all_configs.items():
        for split, data_path in config["paths"].items():
            config["paths"][split] = os.path.join(output_base_path, "data", data_path)

        py_io.write_json(
            data=config,
            path=os.path.join(output_base_path, "configs", f"{task_name}.json"),
        )


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    output_base_path = zconf.attr(type=str)


def main():
    args = RunConfiguration.default_run_cli()
    write_data_configs(output_base_path=args.output_base_path)


if __name__ == "__main__":
    main()
