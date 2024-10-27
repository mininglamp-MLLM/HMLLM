# Evaluation

## Evaluation Strategy

The Video-SME dataset includes two main tasks: **Subjectivity Task** and **Objectivity Task**, each with distinct evaluation protocols and testing methods.

| Task Name  | Evaluation Type   | Training Videos | Testing Videos | Training Q&A | Testing Protocol | Testing Q&A          |
|--|--|--|--|--|--|--|
| **1. Subjectivity**  | Multi-classification | 426           | 72             | 145,107      | P1, P2           | 2,640 (P1), 26,724 (P2) |
| **2. Objectivity**   | Text generation   | 426             | 72             | 5,762        | --               | 954                   |

* **Subjectivity Task**: Evaluated through multi-classification based on Subjective Response Indicators (SRI) and audience demographics. Protocol P1 targets a broad audience, while Protocol P2 extends P1 by integrating SRIs for specific audience profiles.
* **Objectivity Task**: Evaluated using text generation based on GPT-3.5, generating objective analyses of advertisement content. For more details on quantitative evaluation, refer to this [GitHub project](https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main/quantitative_evaluation).

## Running Inference Code
All inference is based on OpenAI-style interfaces and is directly executed using the trained large model.

As such, this part of the code does not cover the inference process for specific models. It is recommended to use inference libraries like VLLM, which abstract several interface details to simplify adaptation across different models.

## Evaluation Commands

### Subjectivity Task
Run the following command to evaluate the subjectivity task:
```sh
sh eval_task1.sh
```

### Objectivity Task
Run the following command to evaluate the objectivity task:
```sh
sh eval_task2.sh
```
