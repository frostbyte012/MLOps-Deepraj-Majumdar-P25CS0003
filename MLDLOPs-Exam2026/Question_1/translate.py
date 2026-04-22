from transformers import pipeline

# Load from the local folder we just cloned
pipe = pipeline("translation", model="./opus-mt-bn-en")

def translate_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    translations = []
    for line in lines:
        clean_line = line.strip()
        if clean_line:
            # Add truncation=True to handle the long 658-token line
            result = pipe(clean_line, truncation=True, max_length=512)
            translations.append(result[0]['translation_text'])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for trans in translations:
            f.write(trans + '\n')

    print(f"Translation complete. Saved to {output_path}")
    if translations:
        print(f"\n--- FOR YOUR EXAM Q1.6 ---")
        print(f"First statement output: {translations[0]}")
        print(f"--------------------------\n")

if __name__ == "__main__":
    translate_file("input.txt", "output.txt")