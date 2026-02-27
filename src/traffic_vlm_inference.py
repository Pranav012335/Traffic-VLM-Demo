import os
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def load_model():
    print("Loading model Qwen/Qwen2-VL-2B-Instruct...")
    # Load the model in half-precision on the available device(s)
    # Using device_map="auto" will automatically distribute the model across available GPUs
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return model, processor

def process_images(model, processor, images_dir):
    if not os.path.exists(images_dir):
        print(f"Directory {images_dir} does not exist. Please create it and add images.")
        return

    import mimetypes
    image_files = []
    for f in os.listdir(images_dir):
        if os.path.isfile(os.path.join(images_dir, f)):
            mime_type, _ = mimetypes.guess_type(f)
            if mime_type and mime_type.startswith('image/'):
                image_files.append(f)

    print(f"Found {len(image_files)} images in {images_dir}. Processing...")
    
    # Create an output directory
    output_dir = os.path.join(os.path.dirname(images_dir), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        
        txt_filename = os.path.splitext(image_file)[0] + ".txt"
        txt_filepath = os.path.join(output_dir, txt_filename)
        if os.path.exists(txt_filepath):
            print(f"Skipping {image_file}, already processed.")
            continue

        print(f"\n--- Processing {image_file} ---")
        
        try:
             # Using qwen_vl_utils format as recommended for Qwen2-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": "Describe the traffic conditions in this image, including any vehicles, pedestrians, or road hazards."},
                    ],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to the appropriate device (cuda if available)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = inputs.to(device)

            # Inference
            print("Generating description...")
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            print(f"Description:\n{output_text[0]}")
            
            # Save the description to a text file
            txt_filename = os.path.splitext(image_file)[0] + ".txt"
            txt_filepath = os.path.join(output_dir, txt_filename)
            with open(txt_filepath, "w", encoding="utf-8") as f:
                f.write(output_text[0])
            print(f"Saved description to: {txt_filepath}")
            
            # Move the processed image to a completed folder
            import shutil
            completed_dir = os.path.join(os.path.dirname(images_dir), "completed")
            os.makedirs(completed_dir, exist_ok=True)
            shutil.move(image_path, os.path.join(completed_dir, image_file))
            print(f"Moved `{image_file}` to: {completed_dir}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    # Point to the data/inputs directory using robust relative paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "data", "inputs")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")
        print("Please place your traffic images in this directory and run the script again.")
    else:
        model, processor = load_model()
        process_images(model, processor, target_dir)
