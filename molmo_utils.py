def load_model():
    return model, processor

def do_inference(image, prompt, temperature=0.2):
    return output_text