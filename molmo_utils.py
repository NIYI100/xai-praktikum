from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torch
import re
from PIL import Image


def load_model(model_name):
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto'
    )
    
    return model, processor

def do_inference(image, prompt, model, processor, temperature=0.2):
    inputs = processor.process(
        images=image,
        text=prompt
    )
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # Generate output
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>", temperature=temperature, do_sample=True),
                tokenizer=processor.tokenizer,
                return_dict_in_generate=True,
                output_logits=True
            )
    output_text = _generate_output_text(output, inputs, processor)
    return output_text, output, inputs

    
def get_coordinates(output_text, width, height):
    # Pattern matches two floats (including optional signs) separated by a comma
    pattern = r"\(([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\)"
    match = re.search(pattern, output_text)
    if match:
        # Convert captured strings to floats
        x, y = map(float, match.groups())
        return _scale_coordinates(x, y, width, height)
    else:
        return (-1, -1)

def refactor_coordinates_to_where_to_place(list_of_coordinates):
    object_to_move = []
    where_to_place = []
    
    for coordinates in list_of_coordinates:
        object_to_move.append(coordinates[0])
        where_to_place.append(coordinates[1])
    return [object_to_move, where_to_place]

def calculate_probability_of_coordinates(output, inputs, processor, ignore_decimal_places=True):
    softmax_fn = torch.nn.Softmax(dim=-1)

    logits = output.logits
    generated_logits = torch.cat(logits, dim=0)
    generated_tokens = output.sequences[0, inputs['input_ids'].size(1):]

    #coord_1_index, coord_2_index = _get_indeces_of_coords(generated_tokens, processor, inputs, ignore_decimal_places)
    coord_1_logits = generated_logits[1:3] + generated_logits[7:9]
    #coord_2_logits = generated_logits[13:15] + generated_logits[19:21]
    
    logit_probs_1 = softmax_fn(coord_1_logits)
    #logit_probs_2 = softmax_fn(coord_2_logits)
    
    highest_probs_1, _ = torch.max(logit_probs_1, dim=-1)
    #highest_probs_2, _ = torch.max(logit_probs_2, dim=-1)

    output_probs_1 = torch.prod(highest_probs_1)
    #output_probs_2 = torch.prod(highest_probs_2)

    return output_probs_1.item()#, output_probs_2.item()


def _get_indeces_of_coords(generated_tokens, processor, inputs, ignore_decimal_places):
    index_of_coords = []
    for i, token in enumerate(generated_tokens):
        char = processor.tokenizer.decode(token, skip_special_tokens=True)
        print(f"{char}, {i}")
        # Start of Coordinate
        if("(" in char):
            index_of_coords.append(i + 1)
        # We only want the int
        if (ignore_decimal_places):
            if("." in char):
                index_of_coords.append(i)
        # We want the whole coordinate
        else:
            if (")" in char):
                index_of_coords.append(i)
    return index_of_coords[0:2], index_of_coords[2:4]

def _scale_coordinates(x, y, width, height):
    scaled_x = float(x) / 100 * width
    scaled_y = float(y) / 100 * height
    return (int(scaled_x), int(scaled_y))

def _generate_output_text(output, inputs, processor):
    # Access generated sequences
    generated_tokens = output.sequences[0, inputs['input_ids'].size(1):]
    # Decode the generated tokens
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text