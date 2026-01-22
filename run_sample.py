import sys
import os
from dotenv import load_dotenv
load_dotenv()

from synapta_segmenter import Pipeline

def main():
    pdf_path = "data/Investments.pdf"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "chapter9_segments.json")
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return

    pipeline = Pipeline()
    pipeline.run(pdf_path, output_path, page_range=(312, 343))

if __name__ == "__main__":
    main()
