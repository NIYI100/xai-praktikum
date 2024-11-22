from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch

# load the processor
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    #torch_dtype='auto',
    torch_dtype=torch.float16,
    device_map='auto'
)

# process the image and text
inputs = processor.process(
    images=[Image.open("im_11.jpg")],
    text="Point to the white basket"
)

# move inputs to the correct device and make a batch of size 1
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
print(model.device)

# generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
  output = model.generate_from_batch(
      inputs,
      GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
      tokenizer=processor.tokenizer
  )


# only get generated tokens; decode them to text
generated_tokens = output[0,inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

# print the generated text
print(generated_text)