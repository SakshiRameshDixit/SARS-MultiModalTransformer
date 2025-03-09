import os
import csv
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch

# Set device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the Mini-InternVL model
path = "OpenGVLab/Mini-InternVL-Chat-2B-V1-5"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
image_processor = CLIPImageProcessor.from_pretrained(path)

# Function to process an image
def process_image(image_path):
    if not os.path.exists(image_path):
        return None
    try:
        image = Image.open(image_path).resize((448, 448))
        pixel_values = image_processor(images=image, return_tensors='pt').pixel_values.to(torch.float16).to(device)
        return pixel_values
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to interpret the model's response as "yes" or "no"
def interpret_response(response):
    response = response.strip().lower()
    if 'yes' in response:
        return 'yes'
    elif 'no' in response:
        return 'no'
    else:
        return 'no'

# Calculate similarity based on model answers to questions with updated weights
def calculate_similarity(image1_path, image2_path, questions):
    # Assign weights based on the question index
    weights = [2] * 10 + [1] * (len(questions) - 10)  # First 10 questions have weight 2, the rest have weight 1
    total_weight = sum(weights)

    # Process the two images
    pixel_values1 = process_image(image1_path)
    pixel_values2 = process_image(image2_path)

    if pixel_values1 is None or pixel_values2 is None:
        return 0, []

    answers1, answers2 = [], []
    for question in questions:
        try:
            # Get model responses for both images
            response1 = model.chat(tokenizer, pixel_values1, question, dict(max_new_tokens=256, do_sample=False))
            response2 = model.chat(tokenizer, pixel_values2, question, dict(max_new_tokens=256, do_sample=False))

            # Interpret responses and store answers
            answers1.append(interpret_response(response1))
            answers2.append(interpret_response(response2))
        except Exception as e:
            print(f"Error with question '{question}': {e}")
            answers1.append("no")
            answers2.append("no")

    # Calculate weighted similarity score
    weighted_score = sum(
        weight if a1 == a2 else 0
        for a1, a2, weight in zip(answers1, answers2, weights)
    )

    # Normalize the score to a percentage (from 0 to 100)
    similarity_percentage = (weighted_score / total_weight) * 100

    return similarity_percentage, answers1
    # Calculate weighted similarity score
    weighted_score = sum(
        weight if a1 == a2 else 0
        for a1, a2, weight in zip(answers1, answers2, weights)
    )

    # Normalize the score to a percentage
    similarity_percentage = (weighted_score / total_weight) * 100

    return similarity_percentage, answers1
# Evaluate similarity for pairs in a CSV file
def evaluate_csv_pairs(input_csv, output_csv, base_dir, questions, max_pairs=5):
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header + ['Similarity'] + [f"Q{i+1}" for i in range(len(questions))])

        pair_count = 0
        for row in reader:
            if max_pairs is not None and pair_count >= max_pairs:
                break

            image1_path = os.path.join(base_dir, row[0])
            image2_path = os.path.join(base_dir, row[1])

            similarity, answers = calculate_similarity(image1_path, image2_path, questions)
            writer.writerow(row + [similarity] + answers)

            print(f"Processed pair {pair_count + 1}: {row[0]} and {row[1]} -> Similarity: {similarity}%")

            pair_count += 1

# Count rows in a CSV file (excluding header)
def count_csv_rows(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        return sum(1 for _ in reader)

# Process all datasets
def process_all_datasets(train_csv, test_csv, val_csv, train_dir, test_dir, val_dir, questions):
    train_pairs = count_csv_rows(train_csv)
    test_pairs = count_csv_rows(test_csv)
    val_pairs = count_csv_rows(val_csv)

    evaluate_csv_pairs(train_csv, 'train_results.csv', train_dir, questions, max_pairs=train_pairs)
    evaluate_csv_pairs(test_csv, 'test_results.csv', test_dir, questions, max_pairs=test_pairs)
    evaluate_csv_pairs(val_csv, 'validation_results.csv', val_dir, questions, max_pairs=val_pairs)

# Paths and questions
train_dir = '/workspace/train/images'
test_dir = '/workspace/test'
val_dir = '/workspace/val/images'

train_csv = '/workspace/csvDataset/train_image_pairs.csv'
test_csv = '/workspace/csvDataset/test_image_pairs.csv'
val_csv = '/workspace/csvDataset/validation_image_pairs.csv'

questions = [
    "Answer 'yes' or 'no' only: Are there any vehicles moving in the opposite direction in the image?",
    "Answer 'yes' or 'no' only: Is the image taken during nighttime (dark environment)?",
    "Answer 'yes' or 'no' only: Does the image depict an urban environment with buildings or city infrastructure?",
    "Answer 'yes' or 'no' only: Is the road visibly congested with a high density of vehicles?",
    "Answer 'yes' or 'no' only: Are there visible bridges or flyovers in the image?",
    "Answer 'yes' or 'no' only: Is there visible construction work or a roadblock in the image?",
    "Answer 'yes' or 'no' only: Is there a visible pedestrian crossing or zebra crossing in the image?",
    "Answer 'yes' or 'no' only: Are there visible speed limit signs in the image?",
    "Answer 'yes' or 'no' only: Is there a visible emergency vehicle (e.g., ambulance, fire truck, or police car) in the image?",
    "Answer 'yes' or 'no' only: Are there visible toll booths or entry gates in the image?",
    "Answer 'yes' or 'no' only: Is there visible snowfall or icy conditions on the road?",
    "Answer 'yes' or 'no' only: Are there visible traffic lights in the image?",
    "Answer 'yes' or 'no' only: Are there visible two-wheeled vehicles (motorcycles or bicycles) on the road in the image?",
    "Answer 'yes' or 'no' only: Is the road visible in the image a highway or expressway?",
    "Answer 'yes' or 'no' only: Are there visible road markings (e.g., lane markers, arrows) in the image?",
    "Answer 'yes' or 'no' only: Is the visibility in the image low due to fog or smoke?",
    "Answer 'yes' or 'no' only: Is the road surface dry in the image?",
    "Answer 'yes' or 'no' only: Are there visible vehicles with their headlights on in the image?",
    "Answer 'yes' or 'no' only: Are there visible pedestrians on the road in the image?",
    "Answer 'yes' or 'no' only: Is there a visible detour or alternative route sign in the image?",
    "Answer 'yes' or 'no' only: Is there a visible pedestrian crossing or zebra crossing in the image?"
]


# Process datasets
process_all_datasets(train_csv, test_csv, val_csv, train_dir, test_dir, val_dir, questions)