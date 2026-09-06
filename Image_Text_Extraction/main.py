import google.generativeai as genai
import mimetypes
import json
import os
from pathlib import Path
import time


class RateLimiter:
    def __init__(self, max_rpm=5):
        self.max_rpm = max_rpm
        self.request_count = 0
        self.window_start = time.time()

    def make_request(self):
        current_time = time.time()

        # Reset counter if more than 60 seconds have passed
        if current_time - self.window_start >= 60:
            self.request_count = 0
            self.window_start = current_time
            print("🔄 RPM counter reset")

        # Check if we've exceeded the limit
        if self.request_count >= self.max_rpm:
            wait_time = 60 - (current_time - self.window_start)
            if wait_time > 0:
                print(f"⏳ RPM limit reached ({self.request_count}/{self.max_rpm}). Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                # Reset after waiting
                self.request_count = 0
                self.window_start = time.time()

        # Make the request
        self.request_count += 1
        print(f"📊 Request {self.request_count}/{self.max_rpm} in current minute")


rate_limiter = RateLimiter(max_rpm=4)


def extract_json_from_response(response_text):
    """
    Extract JSON from response text, removing markdown code blocks and handling escaped characters
    """
    try:
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]  # Remove ```json
        if response_text.startswith("```"):
            response_text = response_text[3:]  # Remove ```
        if response_text.endswith("```"):
            response_text = response_text[:-3]  # Remove closing ```

        # Remove any leading/trailing whitespace
        response_text = response_text.strip()

        # Parse the JSON
        parsed_json = json.loads(response_text)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response text: {response_text}")
        # Return a structured error in the expected format
        return {
            "position_analysis": {
                "location_identifiers": [],
                "reference_objects": []
            },
            "detailed_description": "خطا در پردازش تصویر - قالب JSON نامعتبر"
        }
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return {
            "position_analysis": {
                "location_identifiers": [],
                "reference_objects": []
            },
            "detailed_description": "خطا در پردازش تصویر"
        }


def analyze_image_folder_with_resume(folder_path, api_key, output_json="image_descriptions_v01.json", resume=True):
    """
    Analyze images with resume capability and progress tracking
    """
    rate_limiter.make_request()

    supported_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        folder = Path(folder_path)
        image_files = [f for f in folder.iterdir()
                       if f.is_file() and f.suffix.lower() in supported_extensions]

        if not image_files:
            print(f"No image files found in {folder_path}")
            return

        # Load existing descriptions if resuming
        existing_descriptions = {}
        if resume and os.path.exists(output_json):
            with open(output_json, 'r', encoding='utf-8') as f:
                existing_descriptions = json.load(f)
            print(f"Resuming from existing file with {len(existing_descriptions)} processed images")

        descriptions = existing_descriptions.copy()
        processed_count = len(existing_descriptions)

        print(f"Found {len(image_files)} total images ({processed_count} already processed)")

        for i, image_path in enumerate(image_files, 1):
            # Skip if already processed
            if resume and image_path.name in descriptions:
                continue

            try:
                print(f"Processing {i}/{len(image_files)}: {image_path.name}")

                with open(image_path, 'rb') as file:
                    image_data = file.read()

                mime_type, _ = mimetypes.guess_type(image_path)

                prompt = """شما یک سیستم متخصص موقعیت‌یابی داخلی هستید.تمام تصاویر مربوط به فضای داخلی یک «دانشکده 
                مهندسی دانشگاهی» هستند. فقط از مفاهیم، فضاها و نشانه‌هایی استفاده کن که به‌طور منطقی در یک دانشکده 
                مهندسی وجود دارند. تصویر را تحلیل کنید و فقط اطلاعات ضروری و ثابت محیط را استخراج کنید. 

                     موارد مورد نیاز:
                    1. شناسه‌های مکانی: شماره اتاق، کدها (A310، A3 و ...)، متن تابلوها، نام بخش‌ها.
                    2. ساختار فضا: نوع فضا و موقعیت نسبی (راهرو، تقاطع، ابتدا/انتهای راهرو).
                    3. عناصر ثابت مناسب ترکینگ: درها، دیوارها، تابلوهای ثابت، ستون‌ها و سایر اجزای غیرمتحرک.

                     ممنوع:  
                    - توصیف رنگ‌ها، جنس‌ها، نورپردازی، زیبایی‌شناسی، کف‌پوش، بافت‌ها  
                    - جملات طولانی یا روایت‌گونه  
                    - هرگونه جزئیات غیرکاربردی برای موقعیت‌یابی
                    - استفاده از هر زبان غیر از فارسی

                     فقط اطلاعات کاربردی و فشرده ارائه شود.

                     خروجی دقیقاً در قالب JSON زیر باشد:

                    {
                      "position_analysis": {
                        "location_identifiers": ["<شناسه‌های مکانی استخراج‌شده>"],
                        "reference_objects": [
                          {
                            "object_name": "<نام شیء ثابت>",
                            "object_type": "<نوع، مثل در/دیوار/تابلو>",
                            "relative_position": "<جلو، چپ، راست، پشت>",
                            "suitability_for_distance_estimation": "<بله/خیر>"
                          }
                        ]
                      },
                    "detailed_description":  توضیح بسیار کوتاه (حداکثر یک جمله) شامل: نوع فضا + مهم‌ترین نشانه‌های ثابت + نسبت مکانی اشیاء به یکدیگر (مثلاً نزدیک، روبه‌رو، سمت چپ/راست، کنار). از توضیحات غیرضروری مانند رنگ، جنس یا زیبایی‌شناسی اجتناب کن."
                    }

                     **detailed_description باید حداکثر یک جمله کوتاه و حتما به زبان فارسی باشد (20 کلمه یا کمتر).
                """

                response = model.generate_content([
                    {
                        'mime_type': mime_type,
                        'data': image_data
                    },
                    prompt
                ])

                # Extract and parse JSON from the response
                parsed_json = extract_json_from_response(response.text)

                descriptions[image_path.name] = {
                    'position_analysis': parsed_json.get('position_analysis', {}),
                    'detailed_description': parsed_json.get('detailed_description', ''),
                    'file_path': str(image_path),
                    'file_size': len(image_data),
                    'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }

                print(f"✓ Completed: {image_path.name}")
                print(
                    f"Location identifiers: {parsed_json.get('position_analysis', {}).get('location_identifiers', [])}")
                print(
                    f"Reference objects: {len(parsed_json.get('position_analysis', {}).get('reference_objects', []))} objects")

                # Save progress after each image
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(descriptions, f, indent=2, ensure_ascii=False)

                # Small delay to avoid rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"✗ Error processing {image_path.name}: {e}")
                break

        print(f"\n✅ Successfully processed {len(descriptions)} images!")
        print(f"📁 Results saved to: {output_json}")

        return descriptions

    except Exception as e:
        print(f"General error: {e}")
        return None


def load_api_keys(file_path="API_keys.txt"):
    """
    Load API keys from a file (one key per line)
    """
    try:
        with open(file_path, 'r') as f:
            # Read keys, strip whitespace, and filter out empty lines
            keys = [line.strip() for line in f if line.strip()]
        print(f"✅ Loaded {len(keys)} API keys from {file_path}")
        return keys
    except FileNotFoundError:
        print(f"❌ API keys file not found: {file_path}")
        return []


# Usage
API_KEYs = load_api_keys()
FOLDER_PATH = "images_test"  # Your folder with images
OUTPUT_FILE = "image_test_descriptions.json"

for i, api_key in enumerate(API_KEYs):
    print(f"API-{i}")
    # Process all images
    descriptions = analyze_image_folder_with_resume(FOLDER_PATH, api_key, OUTPUT_FILE)

    if descriptions:
        print(f"\nTotal images processed: {len(descriptions)}")
