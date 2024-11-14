
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

image_path = r'images/Capture6.PNG'
with Image.open(image_path) as img:
    img = img.resize((1024, 1024))
    byte_array = io.BytesIO()
    img.save(byte_array, format="PNG")
    img_bytes = byte_array.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')



promt = '''
Analyze the provided image to detect all visible commercial or public facility signboards that explicitly represent a Point of Interest (POI), such as a business, landmark, or organization. Exclude any traffic, directional, cautionary, or regulatory signs (e.g., 'STOP,' 'SLOW DOWN,' 'YIELD,' 'NO PARKING'). For each detected POI signboard, strictly extract only the following details if visibly present:

    POI Name: The name or title of the Point of Interest (POI) on the signboard (e.g., a business or place name).
    Address: The full address, including street, locality, city, and postal code, if available.
    Email: Any visible email address.
    Phone 1: Primary phone number, if visible.
    Phone 2: Secondary phone number, if visible.
    Website: Website URL, if visible.

Structure the extracted details for each relevant POI signboard in a JSON response. If a detail is not visible on a signboard, leave it as an empty string. If a phone number is blurred or partially visible, leave it as an empty string rather than making assumptions. Below is a JSON object formatted as follows:

{
  "signboard_id": 1,
  "poi_name": "POI Name",
  "address": "Street, Sublocality, Locality, City, Postal Code",
  "email": "email@example.com",
  "phone_1": "",
  "phone_2": "",
  "website": "https://www.example.com"
}

Return only the JSON response without explanations, ignoring any detected signboards that do not represent a POI.
'''
max_gen_len = 1024

temperature = 0

top_p = 1


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
    # print("Response:", response_body['generation'])
    # print("Type:", type(response_body['generation']))
    generation_response = response_body['generation']

    # generation_response = complete_json_string(generation_response)

    print('Json Output',generation_response)
    print('break')
    # print('Json Output type ',type(generation_response))

    # json_match = re.search(r'\[.*\]', generation_response, re.DOTALL)

    json_match = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", generation_response)
    pattern = r'''
        \[
        \s*(\{
        \s*"signboard_id":\s*\d+,\s*
        "poi_name":\s*"[^"]*",\s*
        "address":\s*"[^"]*",\s*
        "email":\s*"[^"]*",\s*
        "phone_1":\s*"[^"]*",\s*
        "phone_2":\s*"[^"]*",\s*
        "website":\s*"[^"]*"\s*\}
        (?:,\s*\{
        \s*"signboard_id":\s*\d+,\s*
        "poi_name":\s*"[^"]*",\s*
        "address":\s*"[^"]*",\s*
        "email":\s*"[^"]*",\s*
        "phone_1":\s*"[^"]*",\s*
        "phone_2":\s*"[^"]*",\s*
        "website":\s*"[^"]*"\s*\})*
        \s*\]
        '''

    if json_match: 
        json_data = json_match.group(1) # Extract the JSON string within the markers 
        try: 
            signboards = json.loads(json_data)
            # Output the parsed JSON in a well-structured format
            print(json.dumps(signboards, indent=2))
            print("successful to decode JSON1.")
        except json.JSONDecodeError:
            print("Failed to decode JSON.")
    # elif generation_response.strip().startswith('[') and generation_response.strip().endswith(']'):
    #     try:
    #         signboards = json.loads(generation_response.strip())  # Parse the raw JSON directly
    #         # print("Json Output:", json.dumps(signboards, indent=2))  # Output the parsed JSON in a well-structured format
    #         print("successful to decode JSON2.")
    #     except json.JSONDecodeError:
    #         print("Failed to decode JSON.")
    elif re.search(pattern, generation_response, re.VERBOSE):
        print("successful to decode JSON2.")
    else: 
        print("No JSON data found.")

    # if json_match:
    #     json_content = json_match.group(0)
    #     try:
    #         # Parse the extracted JSON content
    #         json_output = json.loads(json_content)
    #         print(json_output)  # This will be a properly formatted dictionary/list
    #     except json.JSONDecodeError as e:
    #         print("Failed to parse JSON:", e)
    # else:
    #     print("No JSON content found in the output.")

    # # Attempt to parse JSON
    # try:
    #     response_data = json.loads(generation_response)
    #     print("Parsed JSON:", response_data)
    # except json.JSONDecodeError as e:
    #     print("JSON parsing error:", e)
    #     print("Faulty JSON response:", generation_response)

    # # Deduplication logic if necessary
    # unique_data = {}
    # for entry in response_data:
    #     poi_name = entry.get("poi_name")
    #     if poi_name not in unique_data:
    #         unique_data[poi_name] = entry

    # # Convert deduplicated data back to a list
    # unique_response_data = list(unique_data.values())
    # print("Final JSON data:", json.dumps(unique_response_data, indent=2))

except client.exceptions.ValidationException as e:
    print("Validation error:", e)