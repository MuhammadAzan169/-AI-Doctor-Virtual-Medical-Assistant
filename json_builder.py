import re
import json
from typing import List, Dict

def extract_prescription_data(text: str) -> Dict:
    def extract_field(pattern, source=None, default=""):
        src = source if source else text
        match = re.search(pattern, src, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else default

    def extract_medications(section: str) -> List[Dict]:
        if not section or "Not applicable" in section or "No medication prescribed" in section:
            return []

        meds = []
        med_blocks = re.split(r"(?:\n|^)\s*(?:\d+[\.\)]|\-|\*)\s*(?=\*\*Name\*\*:)", section.strip())

        for block in med_blocks:
            block = block.strip()
            if not block:
                continue

            if not block.startswith("**Name**:"):
                block = "**Name**: " + block

            raw_name = extract_field(r"\*\*Name\*\*:\s*(.+?)(\n|$)", block)
            cleaned_name = re.sub(r"^\d+[\.\)]\s*", "", raw_name).strip()

            dosage = extract_field(r"\*\*Dosage and Route\*\*:\s*(.+?)(\n|$)", block)
            freq = extract_field(r"\*\*Frequency and Duration\*\*:\s*(.+?)(\n|$)", block)
            refills = extract_field(r"\*\*Refills\*\*:\s*(.+?)(\n|$)", block)

            instr_match = re.search(r"\*\*(?:Special Instructions|Special Instructions or Warnings)\*\*:\s*(.+?)(\n|$)", block)
            instructions = instr_match.group(1).strip() if instr_match else ""

            meds.append({
                "name": cleaned_name,
                "brand_names": [],
                "dosage_and_route": dosage,
                "frequency_and_duration": freq,
                "refills": refills,
                "special_instructions": instructions
            })

        return meds




    def extract_non_pharm_recommendations(section: str) -> List[Dict]:
        if not section:
            return []
        recommendations = []
        lines = section.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(?:\d+[\.\)]|\-|\*)\s*(\*\*(.+?)\*\*:)?\s*(.+)", line)
            if match:
                title = match.group(2).strip() if match.group(2) else match.group(3).strip()
                detail = match.group(3).strip()
                recommendations.append({
                    "title": title,
                    "details": {"text": detail}
                })
        return recommendations

    def extract_medical_tests(section: str) -> List[Dict]:
        if not section:
            return []
        tests = []
        lines = section.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(?:\d+[\.\)]|\-|\*)\s*(\*\*(.+?)\*\*:)?\s*(.+)", line)
            if match:
                title = match.group(2).strip() if match.group(2) else match.group(3).strip()
                detail = match.group(3).strip()
                tests.append({
                    "test_name": title,
                    "details": {"text": detail}
                })
        return tests

    patient_info = {
        "name": extract_field(r"\*\*Patient Information\*\*:\s*([\w\s]+),"),
        "age": int(extract_field(r",\s*(\d+)\s*years? old", default="0")),
        "gender": extract_field(r"Gender:\s*([A-Za-z]+)"),
        "date": extract_field(r"\*\*Date\*\*:\s*(.+?)(?=\n\*\*|$)", default="")
    }

    diagnosis = extract_field(r"\*\*Diagnosis\*\*:\s*(.*?)(?=\n\*\*|$)")

    med_match = re.search(
        r"\*\*Medication\*\*:?(.*?)(\*\*Non-Pharmacological Recommendations\*\*|\*\*Medical Tests Recommended\*\*|\*\*Follow-Up\*\*|\*\*Prescriber\*\*|$)",
        text,
        re.DOTALL
    )
    medications = extract_medications(med_match.group(1)) if med_match else []

    non_pharm_match = re.search(
        r"\*\*Non-Pharmacological Recommendations\*\*:?(.*?)(\*\*Medical Tests Recommended\*\*|\*\*Follow-Up\*\*|\*\*Prescriber\*\*|$)",
        text,
        re.DOTALL
    )
    non_pharm_recs = extract_non_pharm_recommendations(non_pharm_match.group(1)) if non_pharm_match else []

    test_match = re.search(
        r"\*\*Medical Tests Recommended\*\*:?(.*?)(\*\*Follow-Up\*\*|\*\*Prescriber\*\*|$)",
        text,
        re.DOTALL
    )
    medical_tests = extract_medical_tests(test_match.group(1)) if test_match else []

    prescriber = extract_field(r"\*\*Prescriber\*\*:\s*(.+?)(?=\n|$)").rstrip("-").strip()

    return {
        "patient_info": patient_info,
        "diagnosis": diagnosis,
        "medication": medications,
        "non_pharmacological_recommendations": non_pharm_recs,
        "medical_tests": medical_tests,
        "prescriber": {
            "name": prescriber
        }
    }