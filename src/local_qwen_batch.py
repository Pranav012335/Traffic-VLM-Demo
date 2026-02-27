import os
import shutil
import mimetypes
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

def _is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def load_model(checkpoint_path="Qwen/Qwen2-VL-2B-Instruct"):
    print(f"Loading model {checkpoint_path}...")
    # Leveraging the robust model loading from web_demo_mm.py
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint_path, 
        device_map=device_map,
        torch_dtype="auto"
    )
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    return model, processor

def process_directory(model, processor, inputs_dir, outputs_dir, completed_dir):
    if not os.path.exists(inputs_dir):
        print(f"Input directory {inputs_dir} does not exist.")
        return

    # Gather all media files in the inputs directory
    media_files = []
    for f in os.listdir(inputs_dir):
        file_path = os.path.join(inputs_dir, f)
        if os.path.isfile(file_path):
            mime_type, _ = mimetypes.guess_type(f)
            if (mime_type and (mime_type.startswith('image/') or mime_type.startswith('video/'))) or _is_video_file(f):
                media_files.append(f)

    if not media_files:
        print(f"No media files found in {inputs_dir}.")
        return

    print(f"Found {len(media_files)} media files. Starting processing loop...")

    # Define the strict target analysis prompt
    text_prompt = (
        "Analyze this visual evidence and strictly answer the following 4 questions based on what you see:\n"
        "1. Why did the car stop?\n"
        "2. Is this dangerous?\n"
        "3. What will happen next?\n"
        "4. Is this an accident risk?\n"
        "If a question is unanswerable or irrelevant because there is no car or danger, explain why."
    )

    for media_file in media_files:
        media_path = os.path.join(inputs_dir, media_file)
        print(f"\n────────────────────────────────────────────")
        print(f"Processing: {media_file}")
        
        try:
            # 1. Prepare structured Message Payload
            content = []
            if _is_video_file(media_file):
                content.append({'type': 'video', 'video': f'{os.path.abspath(media_path)}'})
            else:
                content.append({'type': 'image', 'image': f'{os.path.abspath(media_path)}'})
                
            content.append({'type': 'text', 'text': text_prompt})
            
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            # 2. Extract inputs cleanly using qwen_vl_utils (Replaces our old manual OpenCV logic)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            # 3. Tokenize standard inputs
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to device to match the model
            inputs = inputs.to(model.device)

            # 4. Generate Output (Inference)
            print("Generating sequence...")
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            
            # Trim the prompt history from the generated tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            result = output_text[0]
            print(f"\nResponse:\n{result}\n")
            
            # 5. Save and Cleanup
            txt_filename = os.path.splitext(media_file)[0] + ".txt"
            txt_filepath = os.path.join(outputs_dir, txt_filename)
            
            with open(txt_filepath, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"✔ Saved text description to: {txt_filepath}")
            
            # Move the completed media file to prevent reprocessing inside a loop over time
            destination = os.path.join(completed_dir, media_file)
            shutil.copy(media_path, destination)
            os.remove(media_path)
            print(f"✔ Moved original media to: {destination}")

        except Exception as e:
            print(f"❌ Error processing {media_file}:\n{e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Path to the entirely new and isolated data folders
    inputs_dir = os.path.join(base_dir, "qwen_repo_data", "inputs")
    outputs_dir = os.path.join(base_dir, "qwen_repo_data", "outputs")
    completed_dir = os.path.join(base_dir, "qwen_repo_data", "completed")
    
    for folder in [inputs_dir, outputs_dir, completed_dir]:
        os.makedirs(folder, exist_ok=True)
        
    print(f"Monitoring folder for inputs: {inputs_dir}")
    
    # Check if there are any files to process
    if not os.listdir(inputs_dir):
        print(f"Please place your image or video files into the '{inputs_dir}' folder and run this script again.")
    else:
        # Spin up model and execute
        model, processor = load_model()
        process_directory(model, processor, inputs_dir, outputs_dir, completed_dir)
