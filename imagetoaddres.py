
from PIL import Image
import boto3
import json
import io
import base64
import re


# Initialize the Bedrock client
client = boto3.client('bedrock-runtime', region_name='us-west-2')

# Load and resize the image (if required for any external preprocessing, but we won’t send it directly)
#image_path = r'D:\BITS M.Tech\Desertation\Project Demo\signboard_detection\Signboard-Detection\Test\3.png'
def complete_json_string(response: str) -> str:
    # Close any open quotes
    response = re.sub(r'"[^"]*$', '"None"', response)
    # Ensure closing braces are added
    if not response.strip().endswith("]"):
        response += "]"
    if not response.startswith("["):
        response = "[" + response
    return response

image_path = r'D:\BITS M.Tech\Desertation\Streetview\Capture3.PNG'
with Image.open(image_path) as img:
    img = img.resize((1024, 1024))
    byte_array = io.BytesIO()
    img.save(byte_array, format="PNG")
    img_bytes = byte_array.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')



promt = '''
Analyze the provided image to detect all visible signboards. For each detected signboard, extract the following details:
POI Name: The name or title of the Point of Interest (POI) on the signboard.
Address: Any visible The full address visible on the signboard, including street, locality, city, and postal code if available.
Email: Any visible email address
Phone 1: Any visible Primary phone numbers
Phone 2: Any visible Secondary phone numbers 
Website: Any visible website URL.
Detect multiple signboards in the image and structure the extracted details for each signboard in a direct JSON response, with each JSON object formatted as follows:
{
  "signboard_id": 1,
  "poi_name": "POI Name",
  "address": "Street,Sublocality,Locality, City, Postal Code",
  "email": "email@example.com",
  "phone_1": "+123456789",
  "phone_2": "+987654321",
  "website": "https://www.example.com"
}

Return only the direct JSON response without any explanations.
'''
max_gen_len = 2048

temperature = 0.5

top_p = 0.9


# Prepare the payload for the request to Amazon Bedrock
payload = {
    "modelId": "arn:aws:bedrock:us-west-2:992382420390:inference-profile/us.meta.llama3-2-90b-instruct-v1:0",
    "contentType": "application/json",
    "accept": "application/json",
    "body": json.dumps({
        "prompt": promt,
        "images": [img_base64],  # Directly pass the base64 string inside a list
        "max_gen_len": max_gen_len,  # Maximum generated length
        "temperature": temperature,  # Sampling temperature
        "top_p": top_p    
            
    })
}

# Make the request to Amazon Bedrock
try:
    response = client.invoke_model(
        modelId=payload['modelId'],
        contentType=payload['contentType'],
        accept=payload['accept'],
        body=payload['body'],
    )

    # Print the response
    response_body = json.loads(response['body'].read().decode('utf-8'))
    # print("Response:", response_body)
    print("Response:", response_body['generation'])
    print("Type:", type(response_body['generation']))
    generation_response = response_body['generation']

    generation_response = complete_json_string(generation_response)

    # Attempt to parse JSON
    try:
        response_data = json.loads(generation_response)
        print("Parsed JSON:", response_data)
    except json.JSONDecodeError as e:
        print("JSON parsing error:", e)
        print("Faulty JSON response:", generation_response)

    # Deduplication logic if necessary
    unique_data = {}
    for entry in response_data:
        poi_name = entry.get("poi_name")
        if poi_name not in unique_data:
            unique_data[poi_name] = entry

    # Convert deduplicated data back to a list
    unique_response_data = list(unique_data.values())
    print("Final JSON data:", json.dumps(unique_response_data, indent=2))

except client.exceptions.ValidationException as e:
    print("Validation error:", e)