import streamlit as st
from openai import OpenAI
import datetime
import json
import re
from typing import List
from pdf_builder import build_pdf
from tts import text_to_speech
from stt import speech_to_text
from json_builder import extract_prescription_data
from ocr import perform_ocr
from mtest_data_parser import extract_text_from_json
from detect_fracture import predict_fracture
import whisper
from multiprocessing import Process

API_KEY = "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"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("medium")

def ask_ai(messages):
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=messages,
        max_tokens=10000,
    )
    return response.choices[0].message.content

def app():
    with st.spinner('Loading voice model...'):
        model = load_whisper_model()

    st.title("🩺 AI Doctor - Virtual Medical Assistant")

    name = st.text_input("Enter your name:")
    age = st.text_input("Enter your age:")
    gender = st.radio("What's your gender?", ["Male", "Female"])
    
    if name and age:
        st.markdown("### Upload Medical Files (optional)")
        xray_image = st.file_uploader("Upload X-ray Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
        test_report = st.file_uploader("Upload Test Report (Image or JSON)", type=["jpg", "jpeg", "png", "json"])

        use_voice = st.radio("Do you want to use voice for communication?", ["Yes", "No"])

        symptom = ""
        if use_voice == "Yes":
            st.write("Please speak your symptoms when ready...")
            if st.button("Start Recording"):
                text_to_speech("Please describe your symptoms. Please speak when told to do so.")
                symptom = speech_to_text(model)
                st.session_state.symptom = symptom
                st.success(f"You said: {symptom}")
        else:
            symptom = st.text_area("Describe your symptoms:")
            st.session_state.symptom = symptom

        if "prescription_ready" not in st.session_state:
            st.session_state.prescription_ready = False

        if st.button("Generate Prescription") and st.session_state.symptom.strip():
            fracture_status = ""
            if xray_image:
                with open("xray_image.jpeg", "wb") as f:
                    f.write(xray_image.read())
                fracture_status = predict_fracture("xray_image.jpeg")
            else:
                fracture_status = "No X-ray image provided."

            inform_user = "User does not know about the findings in the X-ray image. Please inform them about the findings through your questions. Please do not forget to tell user about their status on X-ray. Just respond in one line. Don't ask any question just inform."

            messages = [
                {"role": "system", "content": f"User uploaded an X-ray image. Please strongly consider this when generating prescription. We have detected the following fracture status: {fracture_status}"},
                {"role": "system", "content": inform_user},
                {"role": "system", "content": f"You are a professional AI doctor. Start by asking medical follow-up questions (more than 7 questions) to better understand the patient's condition. (Just asks questions nothing more and all the questions should be separated by new lines and questions are to be asked by speech so make sure to use simple language and be professional) {inform_user}"},
                {"role": "user", "content": f"My name is {name}, I am {age} years old and {gender}. I am experiencing: {st.session_state.symptom}"}
            ]

            ai_response = ask_ai(messages)
            ai_questions = [q.strip() + "?" for q in ai_response.split("?") if q.strip()]

            st.session_state.messages = messages
            st.session_state.questions = ai_questions
            st.session_state.answers = []
            st.session_state.question_index = 0
            st.session_state.prescription_ready = False
            st.rerun()

        if "questions" in st.session_state and not st.session_state.prescription_ready:
            current_index = st.session_state.question_index
            questions = st.session_state.questions
            if current_index < len(questions):
                current_question = questions[current_index]
                st.subheader("🤖 Follow-Up Question")
                st.write(current_question)

                if use_voice == "Yes":
                    text_to_speech(current_question)
                    if st.button("🎤 Record Voice Answer"):
                        answer = speech_to_text(model)
                        st.session_state.voice_answer = answer
                        st.success(f"You said: {answer}")

                    if "voice_answer" in st.session_state and st.session_state.voice_answer:
                        if st.button("Submit Voice Answer"):
                            st.session_state.answers.append(st.session_state.voice_answer)
                            st.session_state.question_index += 1
                            del st.session_state.voice_answer
                            st.rerun()
                else:
                    temp_key = f"temp_answer_{current_index}"
                    if temp_key not in st.session_state:
                        st.session_state[temp_key] = ""

                    st.session_state[temp_key] = st.text_input(
                        "Your answer:", value=st.session_state[temp_key], key=f"q_{current_index}"
                    )

                    if st.button("Submit Answer"):
                        answer = st.session_state[temp_key]
                        st.session_state.answers.append(answer)
                        st.session_state.question_index += 1
                        st.rerun()

            else:
                st.success("✅ All follow-up questions answered.")
                qna = "\n".join([
                    f"Your question: {st.session_state.questions[i]} My answer: {st.session_state.answers[i]}"
                    for i in range(len(st.session_state.answers))
                ])
                st.session_state.messages.append({"role": "user", "content": qna})

                if test_report:
                    test_path = "uploaded_test"
                    if test_report.name.endswith(".json"):
                        with open(f"{test_path}.json", "wb") as f:
                            f.write(test_report.read())
                        test_ocr = extract_text_from_json(f"{test_path}.json")
                    else:
                        with open(f"{test_path}.png", "wb") as f:
                            f.write(test_report.read())
                        perform_ocr(f"{test_path}.png")
                        test_ocr = extract_text_from_json("output/large_res.json")
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"This is the OCR output of the uploaded medical test report. Please consider this information while generating the prescription. The OCR output is as follows:\n\n{test_ocr}"
                    })

                today = datetime.date.today().strftime("%B %d, %Y")
                prescription_prompt = f"""
                Based on the information above, generate a complete medical prescription in the following format. Be professional, use generic medication names when possible, and ensure it's understandable by pharmacists.

                ---
                **Medical Prescription**

                **Patient Information**: {name}, {age} years old, Gender: {gender.capitalize()}
                **Date**: {today}

                **Diagnosis**: [Insert accurate diagnosis based on previous discussion]

                **Medication**:  
                - **Name**: [Generic Name] (Brand Name according to Pakistani Market)  
                - **Dosage and Route**: [e.g., 500mg orally]  
                - **Frequency and Duration**: [e.g., Twice a day for 5 days]  
                - **Refills**: [e.g., None / 1 refill]  
                - **Special Instructions**: [e.g., Take with food, avoid alcohol]

                **Non-Pharmacological Recommendations**

                **Medical Tests Recommended**

                **Reasoning**: Please provide a brief reasoning for the diagnosis and medication choices. (The tone should be simple and undersandable by patient as a normal person)

                **Prescriber**: Dr. AI Medic. MD
                ---
                """
                st.session_state.messages.append({"role": "user", "content": prescription_prompt})
                original_response = ask_ai(st.session_state.messages)

                st.subheader("📄 Final Prescription")
                st.markdown(original_response)

                reasoning_start = original_response.find("**Reasoning**:")
                reasoning_end = original_response.find("**Prescriber**")

                reasoning = original_response[reasoning_start + len("**Reasoning**:"):reasoning_end].strip()

                final_response = original_response[:reasoning_start].rstrip() + "\n\n**Prescriber**: Dr. AI Medic. MD\n---"

                text_to_speech(reasoning.replace('e.g.', 'for example').replace('i.e.', 'that is'))

                data = extract_prescription_data(final_response)
                with open("prescription.json", "w") as f:
                    json.dump(data, f, indent=2)

                prescription_path = build_pdf("prescription.json")

                st.session_state.prescription_ready = True
                st.success("✅ Prescription generated successfully! Below is the download link.")

                with open(prescription_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Prescription PDF",
                        data=f,
                        file_name="prescription.pdf",
                        mime="application/pdf"
                    )

if __name__ == "__main__":
    app()