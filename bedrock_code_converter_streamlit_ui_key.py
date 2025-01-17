import os
import json
import boto3
import zipfile
import tempfile
import streamlit as st
from pathlib import Path

# AWS Bedrock client configuration
AWS_ACCESS_KEY_ID = "test"
AWS_SECRET_ACCESS_KEY = "J3AzUFrNzy+test"
AWS_REGION = "us-east-1"
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# Initialize AWS Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# Streamlit app configuration
st.set_page_config(page_title="Code Converter", layout="centered")
st.title("Code Converter with AWS Bedrock")
st.write("Upload a zip file of code files and convert them to your desired programming language.")

# Function to extract zip files
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

# Function to invoke AWS Bedrock
def invoke_bedrock(prompt):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.75,
        "max_tokens": 400,
        "anthropic_version": "bedrock-2023-05-31",
    }
    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response["body"].read().decode("utf-8"))
        return "".join([item["text"] for item in response_body["content"] if item["type"] == "text"])
    except Exception as e:
        st.error(f"Error invoking Bedrock: {e}")
        return None

# Function to process files and save converted output locally
def process_and_save_files(zip_file, target_language, output_dir):
    with tempfile.TemporaryDirectory() as temp_dir:
        extract_zip(zip_file, temp_dir)
        supported_extensions = ['.py', '.js', '.java', '.cs']
        files = [f for f in os.listdir(temp_dir) if Path(f).suffix in supported_extensions]

        for file_name in files:
            file_path = os.path.join(temp_dir, file_name)
            with open(file_path, "r") as file:
                content = file.read()
                st.info(f"Processing {file_name}...")
                prompt = f"Convert the following code to {target_language}:\n\n{content}"
                result = invoke_bedrock(prompt)
                if result:
                    # Save the converted file locally
                    output_file_path = os.path.join(output_dir, f"{Path(file_name).stem}_converted.{target_language.lower()}")
                    with open(output_file_path, "w") as output_file:
                        output_file.write(result)
                    st.success(f"Converted file saved: {output_file_path}")
                else:
                    st.error(f"Failed to convert {file_name}")

# Main Application Logic
uploaded_file = st.file_uploader("Upload a zip file containing code files", type="zip")
target_language = st.selectbox("Select the target language for conversion:", ["Python", "JavaScript", "Java", "C#"])

if uploaded_file and target_language:
    if st.button("Convert Files"):
        output_dir = Path("converted_files_output")
        output_dir.mkdir(exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False) as tmp_zip:
            tmp_zip.write(uploaded_file.read())
            zip_file_path = tmp_zip.name

        process_and_save_files(zip_file_path, target_language, output_dir)

        # Display completion message and list of converted files
        st.success("All files have been converted and saved locally.")
        st.write(f"Converted files are located in: `{output_dir.resolve()}`")

        # Option to download as a zip
        zip_output_path = output_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_output_path, "w") as zipf:
            for file in output_dir.iterdir():
                zipf.write(file, arcname=file.name)

        with open(zip_output_path, "rb") as zip_file:
            st.download_button(
                label="Download All Converted Files",
                data=zip_file,
                file_name="converted_files.zip",
                mime="application/zip",
            )
