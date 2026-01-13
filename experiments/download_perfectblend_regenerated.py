from datasets import load_dataset
import json


def main():
    dataset = load_dataset("frankleeeee/PerfectBlend-Regenerated-Llama-3.1-8B-Instruct")["train"]

    with open("./cache/dataset/perfectblend_regenerated.jsonl", "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()