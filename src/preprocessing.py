import pandas as pd
import re

def clean_code(code):
    code = re.sub(r"\n", " ", code)
    code = re.sub(r"\s+", " ", code)
    return code

def preprocess(input_file, output_file):
    df = pd.read_csv(input_file)
    df["clean_code"] = df["code"].apply(clean_code)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    preprocess("data/raw/dataset.csv","data/processed/processed_data.csv")
