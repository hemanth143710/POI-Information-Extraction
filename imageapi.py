from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import boto3
import json
import io
import base64
import re
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000","http://10.10.4.22:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize the Bedrock client
client = boto3.client('bedrock-runtime', region_name='us-west-2')

# Define the payload model for the Bedrock request
class BedrockPayload(BaseModel):
    modelId: str
    contentType: str
    accept: str
    body: dict

# Helper function to parse JSON response
def parse_json_response(generation_response: str):
    # print(generation_response)
    # print('Break')
    json_match = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", generation_response)
    json_match2 = re.search(r"(\[[\s\S]*?\])", generation_response)
    json_match3 = re.search(r"```([\s\S]*?)```", generation_response)
    if json_match:
        print('Match1')
        json_data = json_match.group(1)
        try:
            return json.loads(json_data)
        except json.JSONDecodeError:
            raise ValueError("Failed 1 to decode POI Information.")
    elif json_match2:
        print('Match2')
        json_data2 = json_match2.group(1)
        try:
            return json.loads(json_data2)
        except json.JSONDecodeError:
            raise ValueError("Failed 2 to decode POI Information.")
    elif json_match3:
        print('Match3')
        json_data3 = json_match3.group(1)
        
        # Fix: Wrap the matched content in square brackets to create a valid JSON array
        json_data3 = f"[{json_data3}]"
        
        try:
            return json.loads(json_data3)
        except json.JSONDecodeError:
            raise ValueError("Failed 3 to decode POI Information.")
    else:
        print(generation_response)
        print('Break')
        raise ValueError("Failed to decode POI Information.")
    
app.mount("/static", StaticFiles(directory="static"), name="static")

# Helper function to invoke the model for address parsing
def invoke_model(client, model_id, address_string):
    prompt = f"""
    Task: Parse the following address string into the specified components. Provide the output strictly in JSON format with keys for each component.
    Components to output:
    - H.no
    - Hname
    - Street
    - Locality
    - Sublocality
    - Landmark
    - City
    - State
    - Pincode
    Address string: "{address_string}"
    Ensure that if a component is not present, the value should be "None". 
    Do not include any additional explanations or code. Just provide the JSON output.
    Output format example: {{
        "H.no": "value",
        "Hname": "value",
        "Street": "value",
        "Locality": "value",
        "Sublocality": "value",
        "Landmark": "value",
        "City": "value",
        "State": "value",
        "Pincode": "value"
    }}
    """
    
    payload = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }
    
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )
    
    response_body = json.loads(response['body'].read().decode('utf-8'))
    generation_response = response_body.get("generation")
    
    if generation_response:
        # Parse JSON output directly if it's valid JSON
        try:
            return json.loads(generation_response)
        except json.JSONDecodeError:
            # Fall back to regex parsing if JSON decoding fails
            pattern = r'"([^"]+)":\s*"([^"]*)"'
            matches = re.findall(pattern, generation_response)
            return {key: value for key, value in matches}


templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "name": "Hemanth"})

@app.get("/home", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index8.html", {"request": request, "name": "Hemanth"})

# Endpoint to process the uploaded image and detect signboards
@app.post("/detect-signboards")
async def detect_signboards(image: UploadFile = File(...)):
    try:
        # Load and resize the image
        image_content = await image.read()
        img = Image.open(io.BytesIO(image_content))
        img = img.resize((1024, 1024))
        
        # Convert image to base64
        byte_array = io.BytesIO()
        img.save(byte_array, format="PNG")
        img_bytes = byte_array.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Prepare the prompt
    
        prompt = '''
        Analyze the provided image to detect all visible commercial or public facility signboards that explicitly represent a Point of Interest (POI), such as a business, landmark, or organization. Exclude any traffic, directional, cautionary, or regulatory signs (e.g., 'STOP,' 'SLOW DOWN,' 'YIELD,' 'NO PARKING'). For each detected POI signboard, strictly extract only the following details if visibly present:

        POI Name: The name or title of the Point of Interest (POI) on the signboard (e.g., a business or place name).
        Address: The full address, including street, locality, city, and postal code, if available.
        Email: Any visible email address.
        Phone 1: Primary phone number, if visible.
        Phone 2: Secondary phone number, if visible.
        Website: Website URL, if visible.

        Structure the extracted details for each relevant POI signboard in a JSON response. If a detail is not visible on a signboard, return "None" (as a string) instead of an empty string. If a phone number is blurred or partially visible, leave it as "None" rather than making assumptions. Below is a JSON object formatted as follows:

        {
        "signboard_id": 1,
        "poi_name": "POI Name",
        "address": "Street, Sublocality, Locality, City, Postal Code",
        "email": "email@example.com",
        "phone_1": "None",
        "phone_2": "None",
        "website": "None"
        }

        Return only the JSON response without explanations, ignoring any detected signboards that do not represent a POI.
        '''
        
        # Prepare payload for Bedrock
        payload = {
            "modelId": "arn:aws:bedrock:us-west-2:992382420390:inference-profile/us.meta.llama3-2-90b-instruct-v1:0",
            "contentType": "application/json",
            "accept": "application/json",
            "body": json.dumps({
                "prompt": prompt,
                "images": [img_base64],
                "max_gen_len": 1024,
                "temperature": 0,
                "top_p": 1
            })
        }
        
        # Invoke the model
        response = client.invoke_model(
            modelId=payload['modelId'],
            contentType=payload['contentType'],
            accept=payload['accept'],
            body=payload['body']
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        generation_response = response_body['generation']
        signboards = parse_json_response(generation_response)
        
        return signboards

    except client.exceptions.ValidationException as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Endpoint to parse address components
@app.post("/parse-address")
async def parse_address(address_string: str):
    try:
        model_id = 'arn:aws:bedrock:us-west-2:992382420390:inference-profile/us.meta.llama3-2-90b-instruct-v1:0'  # Update with the valid model ID
        parsed_address = invoke_model(client, model_id, address_string)
        return parsed_address
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing address: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("imageapi:app", host="0.0.0.0", port=8000, reload=True)
