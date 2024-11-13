import boto3

import json

import re



# Set your region

REGION_NAME = 'ap-south-1'  # e.g., 'us-west-2'

# Set the correct model ID

model_id = 'meta.llama3-8b-instruct-v1:0'  # Update with the valid model ID



client = boto3.client('bedrock-runtime', region_name=REGION_NAME)



def invoke_model(client, model_id, address_string):

    # Define your prompt for parsing the address

    prompt = f"""

    Task: Parse the following address string into the specified components. Provide the output strictly in JSON format with keys for each component.



    Components to output:

    - POI Name

    - Street

    - Locality

    - Sublocality

    - City

    - State

    - Pincode

    - Category



    Address string: "{address_string}"



    Ensure that if a component is not present, the value should be "None". 

    Do not include any additional explanations or code. Just provide the JSON output.



    Output format example: {{ "POI Name": "value", "Street": "value", "Locality": "value", "Sublocality": "value", "City": "value", "State": "value", "Pincode": "value", "Category": "value" }}

    Give output in json format

    """

    

    max_gen_len = 512

    temperature = 0.5

    top_p = 0.9



    try:

        # Invoke the model

        response = client.invoke_model(

            modelId=model_id,

            contentType="application/json",

            accept="application/json",

            body=json.dumps({

                "prompt": prompt,            # Updated input text

                "max_gen_len": max_gen_len,  # Maximum generated length

                "temperature": temperature,  # Sampling temperature

                "top_p": top_p               # Cumulative probability for sampling

            })

        )



        if 'body' in response:

            response_body = response['body']

            

            # If response body is a byte stream, decode it

            if hasattr(response_body, 'read'):

                result = json.loads(response_body.read().decode('utf-8'))

            else:

                result = json.loads(response_body)

            

            print("Model Output:", result)

            

            # Checking if 'generation' needs to be parsed

            generation_output = result['generation']

            

            if isinstance(generation_output, str):

                try:

                    # Parse the string to JSON

                    parsed_output = json.loads(generation_output)

                    print("Parsed Output:", parsed_output)

                    

                    # Extracting the 'POI Name' value

                    poi_name = parsed_output.get("POI Name", "Not Found")

                    print("POI Name:", poi_name)

                    

                except json.JSONDecodeError:

                    print("Generation output is not valid JSON, printing as a string:")

                    print("Parsed Output:", generation_output)

                    generation = generation_output  # Accessing the generated output

                    print("Generation Output:", generation)

                  

                    pattern = r'"([^"]+)":\s*"([^"]*)"'



                    # Using findall to extract all key-value pairs

                    matches = re.findall(pattern, generation)



                    # Convert matches to a dictionary

                    result_dict = {key: value for key, value in matches}



                    # Print the resulting dictionary

                    print("Extracted JSON:", result_dict)



                    # Optionally, convert to a JSON string

                    json_result = json.dumps(result_dict, indent=4)

                    print("JSON Output:\n", json_result)



                    # Extract individual fields

                    poi_name = result_dict.get("POI Name")

                    street = result_dict.get("Street")

                    locality = result_dict.get("Locality")

                    sublocality = result_dict.get("Sublocality")

                    city = result_dict.get("City")

                    state = result_dict.get("State")

                    pincode = result_dict.get("Pincode")

                    category = result_dict.get("Category")



                    # Print the extracted fields

                    print("\nExtracted Fields:")

                    print("POI Name:", poi_name)

                    print("Street:", street)

                    print("Locality:", locality)

                    print("Sublocality:", sublocality)

                    print("City:", city)

                    print("State:", state)

                    print("Pincode:", pincode)

                    print("Category:", category)



                    extracted_fields = {

                        "POI Name": poi_name,

                        "Street": street,

                        "Locality": locality,

                        "Sublocality": sublocality,

                        "City": city,

                        "State": state,

                        "Pincode": pincode,

                        "Category": category

                    }

                    print('Extracted Fields JSON:',extracted_fields)



            else:

                print("Parsed Output:", generation_output)

        else:

            print("No body found in the response")



        response_body = json.loads(response.get('body').read())  # Reading and parsing the response body

        

    except json.JSONDecodeError as json_err:

        print("Failed to parse JSON response:", json_err)

    except Exception as e:

        print("Error invoking model:", e)



if __name__ == "__main__":

    # Provide the dynamic address string here

    address_string = "Deduce Technologies 27 Cross Road BTM Bangalore"  # Example address

    invoke_model(client, model_id, address_string)
