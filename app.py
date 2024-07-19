
import streamlit as st
import os
import pdf2image
import pytesseract
from PIL import Image
import pandas as pd
import torch
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Initialize YOLOv5 model
def load_yolov5_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        return None
    return model

model = load_yolov5_model()
if model is None:
    st.stop()  # Stop execution if the model fails to load

# SQLAlchemy setup
db_url = 'sqlite:///extracted_text.db'
engine = create_engine(db_url)
Base = declarative_base()

class ExtractedText(Base):
    __tablename__ = 'extracted_text'
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String)
    content = Column(Text)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Function to perform OCR
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    text = ' '.join(text.split())  # Remove extra spaces
    return text

# Function to convert dataframe to text
def convert_df_to_text(df):
    content = ""
    for index, row in df.iterrows():
        content += f"{row['file_name']}\n{row['content']}\n\n"
    return content

# Streamlit interface
st.title("File Upload and Processing")
uploaded_files = st.file_uploader("Upload PDF or Excel files", type=["pdf", "xlsx"], accept_multiple_files=True)
process_button = st.button("Process Files")

# Function to process Excel files
def process_excel(uploaded_file):
    st.write("Processing Excel file...")
    df = pd.read_excel(uploaded_file, sheet_name=None)

    for sheet_name, data in df.items():
        st.write(f"Processing sheet: {sheet_name}")
        content = data.to_string(index=False)
        content = ' '.join(content.split())  # Remove extra spaces
        excel_text = ExtractedText(file_name=uploaded_file.name, content=content)
        session.add(excel_text)
    session.commit()

# Function to process PDF files
def process_pdf(uploaded_file):
    st.write("Processing PDF file...")
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    images = pdf2image.convert_from_path(uploaded_file.name, poppler_path=r'C:\poppler\poppler-24.02.0\Library\bin')
    content = ""
    for image in images:
        text = extract_text_from_image(image)
        content += text + "\n"
    content = ' '.join(content.split())  # Remove extra spaces
    page_text = ExtractedText(file_name=uploaded_file.name, content=content)
    session.add(page_text)
    session.commit()

if process_button:
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.name.split('.')[-1]
            if file_type == 'pdf':
                process_pdf(uploaded_file)
            elif file_type == 'xlsx':
                process_excel(uploaded_file)
        st.success("Files processed successfully. You can download the extracted data below.")

        st.write("Reading data from database...")
        db_df = pd.read_sql('extracted_text', con=engine)
        st.write("Data retrieved from database:")
        st.write(db_df)

        text_content = convert_df_to_text(db_df)

        st.download_button(
            label="Download Extracted Data as Text",
            data=text_content,
            file_name='extracted_data.txt',
            mime='text/plain',
        )
    else:
        st.warning("Please upload files to process.")
