import PyPDF2
import json
import traceback

def parse_file(file):
    if file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except PyPDF2.utils.PdfReadError:
            raise Exception("Error reading the PDF file.")

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    else:
        raise Exception("Unsupported File Format. Only PDF and .Txt Files are supported")

RESPONSE_JSON = {
    "1": {
        "no": "1",
        "question": "question",
        "answer": "answer",
        "correct": "correct answer",
    },
    "2": {
        "no": "2",
        "question": "question",
        "answer": "answer",
        "correct": "correct answer",
    },
    "3": {
        "no": "3",
        "question": "question",
        "answer": "answer",
        "correct": "correct answer",
    },
}

def get_table_data(response_json):
    try:
        # convert the response_json from a str to dict
        response_dict = json.loads(response_json)
        table_data = []

        for key, value in response_dict.items():
            question = value.get("question", "")
            answer = value.get("answer", "")
            correct = value.get("correct", "")

            table_data.append({"Question": question, "Answer": answer, "Correct": correct})

        return table_data
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        return False
