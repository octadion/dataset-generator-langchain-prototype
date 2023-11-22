import json
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import pandas as pd
import traceback
from utils import parse_file, RESPONSE_JSON, get_table_data
load_dotenv()

llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = """
Text:{text}
Anda adalah seorang ahli hukum yang mengkhususkan diri dalam menganalisis artikel hukum. Dengan teks di atas, tugas Anda adalah \
Membuat dataset dari {number} yang berisi pasangan input dan output tentang:
1. Mengidentifikasi konflik atau kontradiksi dalam artikel.
2. Merevisi artikel berdasarkan konten artikel sebelumnya.
3. Menghasilkan draf artikel yang menggabungkan informasi dari artikel sebelumnya.
Pastikan untuk memberikan respons yang rinci dan akurat dan pastikan untuk memformat respons Anda seperti RESPONSE_JSON di bawah ini

### RESPONSE_JSON
{response_json}
"""
legal_task_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "response_json"],
    template=template
)

legal_task_chain = LLMChain(llm=llm, prompt=legal_task_generation_prompt, output_key="legal_tasks", verbose=True)

text_evaluation_template = """
Anda adalah seorang analis hukum yang bertugas mengevaluasi tugas hukum yang dihasilkan. Berikan analisis rinci dan lakukan revisi yang diperlukan.

Tugas Hukum:
{legal_tasks}

Analisis dan Revisi:
"""
legal_task_evaluation_prompt = PromptTemplate(
    input_variables=["legal_tasks"],
    template=text_evaluation_template
)
legal_evaluation_chain = LLMChain(llm=llm, prompt=legal_task_evaluation_prompt, output_key="legal_evaluation", verbose=True)
generate_evaluate_chain = SequentialChain(
    chains=[legal_task_chain, legal_evaluation_chain],
    input_variables=["text", "number", "response_json"],
    output_variables=["legal_tasks", "legal_evaluation"],
    verbose=True,
)
st.title("Legal Task Dataset Generator 📚⚖️")
with st.form("user_inputs"):
    uploaded_file = st.file_uploader("Upload a legal text file")
    dataset_count = st.number_input("Number of datasets to generate", min_value=1, max_value=10)
    # tone = st.text_input("Insert Dataset Tone", max_chars=100, placeholder="formal")
    button = st.form_submit_button("Generate Datasets")
    if button and uploaded_file is not None and dataset_count:
        with st.spinner("Loading..."):
            try:
                text = parse_file(uploaded_file)  # Define your parse_file function
                datasets = []

                for _ in range(dataset_count):
                    with get_openai_callback() as cb:
                        response = generate_evaluate_chain(
                            {
                                "text": text,
                                "number": dataset_count,
                                "response_json": json.dumps(RESPONSE_JSON)
                            }
                        )

                    if isinstance(response, dict):
                        input_output_pair = {
                            "input": text,
                            "output": response["legal_tasks"],
                        }
                        datasets.append(input_output_pair)
                        legal_tasks = response.get("legal_tasks", None)
                        if legal_tasks is not None:
                            table_data = get_table_data(legal_tasks)
                            if table_data is not None:
                                df = pd.DataFrame(table_data)
                                df.index = df.index + 1
                                st.table(df)
                                st.text_area(label="legal_evaluation", value=response["legal_evaluation"])
                            else:
                                st.error("Error in the table data")
                    else:
                        st.error("Error generating dataset")

                # Display generated datasets
                # df = pd.DataFrame(datasets)
                # st.table(df)

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")