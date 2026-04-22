import sacrebleu

def compute_bleu(output_path, reference_path):
    with open(output_path, 'r', encoding='utf-8') as f:
        preds = [line.strip() for line in f.readlines()]
    
    with open(reference_path, 'r', encoding='utf-8') as f:
        refs = [[line.strip() for line in f.readlines()]]

    bleu = sacrebleu.corpus_bleu(preds, refs)
    print(f"BLEU score: {bleu.score}")

if __name__ == "__main__":
    compute_bleu("output.txt", "reference.txt")