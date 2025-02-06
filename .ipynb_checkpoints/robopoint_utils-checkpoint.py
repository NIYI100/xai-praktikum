from robopoint.model.builder import load_pretrained_model
from robopoint.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from robopoint.conversation import conv_templates, SeparatorStyle

from PIL import Image
from io import BytesIO

import requests
import torch

import matplotlib.pyplot as plt
import math

import os
import re

PROMPT_OUTPUT_FORMAT="Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image. Output 3 coordinates."
PROMPT_START="Locate several points within the vacant space for the following task: "

def load_model(model_path="wentao-yuan/robopoint-v1-vicuna-v1.5-13b"):
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        #load_8bit=True,
        device="cuda"
    )
    return model, processor, tokenizer

def generate_prompt(prompt_objective, prompt_start=PROMPT_START, prompt_output_format=PROMPT_OUTPUT_FORMAT):
    text = '<image>' + '\n' + prompt_start + prompt_objective + ". " + prompt_output_format
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def do_inference(image, prompt, model, processor, tokenizer, temperature=0.2):
    # prepare inputs for the model
    input_ids = tokenizer_image_token(prompt, tokenizer, -200, return_tensors='pt').unsqueeze(0).cuda()
    
    # prepare image input
    image_tensor = process_images([image], processor, model.config)[0].unsqueeze(0).half().cuda()
    
    # autoregressively generate text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            #image_sizes=[image.size],
            do_sample=True,
            temperature=temperature,
            max_new_tokens=256,
            use_cache=True)
    
    # only get generated tokens; decode them to text
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    return output


def do_inference_with_logits(image, prompt, model, processor, tokenizer, temperature=0.2):
    # prepare inputs for the model
    input_ids = tokenizer_image_token(prompt, tokenizer, -200, return_tensors='pt').unsqueeze(0).cuda()
    
    # prepare image input
    image_tensor = process_images([image], processor, model.config)[0].unsqueeze(0).half().cuda()
    
    # autoregressively generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            #image_sizes=[image.size],
            do_sample=True,
            temperature=temperature,
            max_new_tokens=256,
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=True)

    # Extract generated IDs and scores
    generated_ids = outputs.sequences
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return generated_text, outputs

def calculate_probs_per_coordinate(outputs, ignoreDecimalPlaces = False, coordinate_size = 14):
    generated_ids = outputs.sequences

    logits = outputs.logits
    logits = torch.cat(logits, dim=0)
    softmax_fn = torch.nn.Softmax(dim=-1)

    logit_probs = softmax_fn(logits)
    highest_probs, _ = torch.max(logit_probs, dim=-1)

    # Initialize an empty list to store the results
    products = []
    
    # Iterate over logit_probs in chunks of 14
    for i in range(0, len(highest_probs) - 1, coordinate_size):
        if i + coordinate_size > len(highest_probs) - 1:
            break
        chunk = highest_probs[i:i+coordinate_size]  # Get the chunk of 14 elements
        if ignoreDecimalPlaces:
            product = chunk[3] * chunk[10] #Calculate first number for x and y value
        else:
            product = math.prod(chunk)   # Calculate the product of the chunk
        products.append(product)     # Append the result to the products list
        
    return products

def get_coordinates(coord_string, image_width, image_height):
    # Find pattern: (x, y)
    pattern = r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)'
    matches = re.findall(pattern, coord_string)
    
    coordinates = [(float(x), float(y)) for x, y in matches]
    return _scale_coordinates(coordinates, image_width, image_height)

def _scale_coordinates(coordinates, image_width, image_height):
    scaled_coordinates = []
    for x, y in coordinates:
        # Scale coordinates to the image size
        scaled_x = x * image_width
        scaled_y = y * image_height
        scaled_coordinates.append((int(scaled_x), int(scaled_y)))
    return scaled_coordinates