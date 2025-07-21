from paddleocr import PaddleOCR
import json
import os

ocr = PaddleOCR(
    use_angle_cls=False,
    lang='en'
)

def perform_ocr(path_to_test):
    result = ocr.ocr(path_to_test, cls=True)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    json_result = []
    for line in result:
        for word_info in line:
            json_result.append({
                "text": word_info[1][0],
                "confidence": float(word_info[1][1]),
                "box": word_info[0]
            })

    with open(os.path.join(output_dir, "large_res.json"), "w") as f:
        json.dump(json_result, f, indent=2)

    print(f"OCR complete. JSON saved to {output_dir}/large_res.json")
