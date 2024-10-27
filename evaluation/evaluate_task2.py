import os
import argparse
import json
import ast
from multiprocessing import Pool
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
    parser.add_argument("--api_url", default="https://api.openai.com/v1", help="OpenAI URL to call")
    parser.add_argument("--api_key", required=True, help="OpenAI API key.")
    parser.add_argument("--num_tasks", type=int, default=1, help="Number of splits.")
    return parser.parse_args()

def annotate(prediction_set, caption_files, output_dir, api_url, api_key):
    client = OpenAI(base_url=api_url, api_key=api_key)

    for file in caption_files:
        key = file
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['pred']

        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                        {
                            "role": "system",
                            "content":
                                "You are an intelligent chatbot designed for evaluating the contextual understanding of generative outputs for video-based question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine if the generated response aligns with the overall context of the video content. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Evaluate whether the predicted answer aligns with the overall context of the video content. It should not provide information that is out of context or misaligned.\n"
                                "- The predicted answer must capture the main themes and sentiments of the video.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Provide your evaluation of the contextual understanding of the prediction compared to the answer."
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Correct Answer: {answer}\n"
                                f"Predicted Answer: {pred}\n\n"
                                "Provide your evaluation only as a contextual understanding score where the contextual understanding score is an integer value between 0 and 5, with 5 indicating the highest level of contextual understanding. "
                                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is contextual understanding score in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {''score': 4.8}."
                        }
                    ]
            )
            response_message = completion.choices[0].message.content
            response_dict = ast.literal_eval(response_message)
            result_qa_pair = [response_dict, qa_set]

            with open(os.path.join(output_dir, f"{key}.json"), "w") as f:
                json.dump(result_qa_pair, f)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file]

def main():
    args = parse_args()
    
    pred_contents = read_jsonl(args.pred_path)

    video_id_counts = {}
    new_pred_contents = []

    for sample in pred_contents:
        video_id = os.path.basename(sample['video'][0])
        count = video_id_counts.get(video_id, -1) + 1
        video_id_counts[video_id] = count

        new_sample = sample.copy()
        new_sample['video'] += f"_{count}"
        new_pred_contents.append(new_sample)

    id_list = [os.path.basename(x['video'][0]) for x in new_pred_contents]
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    prediction_set = {
        os.path.basename(sample['video'][0]): {"q": sample['question'], "a": sample['answer'], "pred": sample['pred']}
        for sample in new_pred_contents
    }

    incomplete_files = set(id_list) - set(f[:-5] for f in os.listdir(args.output_dir))
    
    while incomplete_files:
        
        num_tasks_to_use = min(len(incomplete_files), args.num_tasks)
        
        part_len = len(incomplete_files) // num_tasks_to_use or 1
        
        all_parts = [
            list(incomplete_files)[i:i + part_len] 
            for i in range(0, len(incomplete_files), part_len)
        ]

        task_args = [(prediction_set, part, args.output_dir, args.api_url, args.api_key) for part in all_parts]

        with Pool() as pool:
            pool.starmap(annotate, task_args) 

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    # Iterate through json files
    for file_name in os.listdir(args.output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(args.output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    # Calculate average score
    score_sum = 0
    count = 0
    for key, result in combined_contents.items():
        count += 1
        score_match = result[0]['score']
        score = int(score_match)
        score_sum += score
    average_score = score_sum / count

    print("Average score for contextual understanding:", average_score)

if __name__ == "__main__":
    main()