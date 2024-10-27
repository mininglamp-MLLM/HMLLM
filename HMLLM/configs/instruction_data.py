import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy

available_corpus = dict(
    # video
    SRI=[
        "/home/HMLLM/dataset/train-EN.json", 
        "/home/HMLLM/dataset/videos",
        "video",
    ],
    # csv
    SRI_pro=[
        "/home/HMLLM/dataset/train-EN-profiles.csv", 
        "/home/HMLLM/dataset/videos",
        "video",
    ],
)
available_corpus["SRI-inst"] = [
    available_corpus["SRI"]
]

available_corpus["SRI-profiles"] = [
    available_corpus["SRI_pro"]
]