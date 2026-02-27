import os
import cv2
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

def process_media(model, processor, inputs_dir):
    if not os.path.exists(inputs_dir):
        print(f"Directory {inputs_dir} does not exist. Please create it and add media.")
        return

    import mimetypes
    media_files = []
    for f in os.listdir(inputs_dir):
        if os.path.isfile(os.path.join(inputs_dir, f)):
            mime_type, _ = mimetypes.guess_type(f)
            if mime_type and (mime_type.startswith('image/') or mime_type.startswith('video/')):
                media_files.append(f)

    print(f"Found {len(media_files)} media files in {inputs_dir}. Processing...")
    
    # Create an output directory
    output_dir = os.path.join(os.path.dirname(inputs_dir), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")

    for media_file in media_files:
        media_path = os.path.join(inputs_dir, media_file)
        
        txt_filename = os.path.splitext(media_file)[0] + ".txt"
        txt_filepath = os.path.join(output_dir, txt_filename)
        if os.path.exists(txt_filepath):
            print(f"Skipping {media_file}, already processed.")
            continue

        print(f"\n--- Processing {media_file} ---")
        
        try:
            mime_type, _ = mimetypes.guess_type(media_file)
            is_video = mime_type and mime_type.startswith('video/')
            
            if is_video:
                # Manually extract frames using OpenCV to bypass qwen_vl_utils video bugs
                cap = cv2.VideoCapture(media_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Sample 4 frames evenly spaced
                sample_frames = 4
                frame_indices = [int(i * frame_count / sample_frames) for i in range(sample_frames)]
                
                video_frames = []
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        # Convert BGR (OpenCV) to RGB (PIL)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        video_frames.append(pil_img)
                cap.release()

                # Instead of sending a video, we send a sequence of images that represent the video
                media_content = []
                for img in video_frames:
                     media_content.append({"type": "image", "image": img})
                     
                text_prompt = (
                    "Analyze this video and strictly answer the following 4 questions based on the visual evidence:\n"
                    "1. Why did the car stop?\n"
                    "2. Is this dangerous?\n"
                    "3. What will happen next?\n"
                    "4. Is this an accident risk?\n"
                    "If a question is unanswerable or irrelevant because there is no car or danger, explain why."
                )
            else:
                media_content = [{"type": "image", "image": f"file://{media_path}"}]
                text_prompt = (
                    "Analyze this image and strictly answer the following 4 questions based on the visual evidence:\n"
                    "1. Why did the car stop?\n"
                    "2. Is this dangerous?\n"
                    "3. What will happen next?\n"
                    "4. Is this an accident risk?\n"
                    "If a question is unanswerable or irrelevant because there is no car or danger, explain why."
                )

             # Using qwen_vl_utils format as recommended for Qwen2-VL
            content_list = []
            content_list.extend(media_content)
            content_list.append({"type": "text", "text": text_prompt})

            messages = [
                {
                    "role": "user",
                    "content": content_list,
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
            txt_filename = os.path.splitext(media_file)[0] + ".txt"
            txt_filepath = os.path.join(output_dir, txt_filename)
            with open(txt_filepath, "w", encoding="utf-8") as f:
                f.write(output_text[0])
            print(f"Saved description to: {txt_filepath}")
            
            # Move the processed image to a completed folder
            import shutil
            completed_dir = os.path.join(os.path.dirname(inputs_dir), "completed")
            os.makedirs(completed_dir, exist_ok=True)
            shutil.move(media_path, os.path.join(completed_dir, media_file))
            print(f"Moved `{media_file}` to: {completed_dir}")
            
        except Exception as e:
            print(f"Error processing {media_file}: {e}")

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
        process_media(model, processor, target_dir)
