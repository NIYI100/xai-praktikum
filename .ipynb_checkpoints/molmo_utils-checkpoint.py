from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torch
import re


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
    return output_text

def get_coordinates(output_text, width, height):
    pattern = r"\(([^,]+), ([^\)]+)\)"
    matches = re.findall(pattern, output_text)
    coordinates = []

    for x, y in matches:
        coordinates.append(_scale_coordinates(x, y, width, height))
    return coordinates

def refactor_coordinates_to_where_to_place(list_of_coordinates):
    object_to_move = []
    where_to_place = []
    
    for coordinates in list_of_coordinates:
        object_to_move.append(coordinates[0])
        where_to_place.append(coordinates[1])
    return [object_to_move, where_to_place]

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