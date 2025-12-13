import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =====================
# CONFIG
# =====================
DATASET_PATH = r"C:\Users\HP\Desktop\PhD\workshop-1\Datasets\smart_building_causal_dataset.csv"
OUTPUT_PATH = r"C:\Users\HP\Desktop\PhD\workshop-1\results\single_agent_results_llama3.csv"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

print("Loading Llama-3-8B-Instruct for single-agent evaluation...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

SINGLE_AGENT_PROMPT = """You are an AI assistant inside a smart building.
Answer the following causal question with a single word: "Yes" or "No",
unless the question explicitly asks for a short textual answer.

Question:
{question}

Answer:
"""

def call_model(prompt: str, max_new_tokens: int = 64) -> str:
    out = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = out[0]["generated_text"]
    return text[len(prompt):].strip()

def normalize_answer(ans: str) -> str:
    ans = ans.strip().lower()
    if "yes" in ans:
        return "Yes"
    if "no" in ans:
        return "No"
    return ans  # for more complex answers in future

def main():
    df = pd.read_csv(DATASET_PATH)
    rows = []

    for idx, row in df.iterrows():
        q = row["question"]
        gt = str(row["answer"]).strip()

        prompt = SINGLE_AGENT_PROMPT.format(question=q)
        model_out = call_model(prompt)
        norm_out = normalize_answer(model_out)

        correct = (norm_out == gt)

        rows.append({
            "question": q,
            "ground_truth": gt,
            "raw_output": model_out,
            "normalized_output": norm_out,
            "correct": int(correct),
            "level": row["level"],
            "scenario": row["scenario"],
        })

        print(f"[{idx+1}/{len(df)}] Correct: {correct} | GT={gt} | OUT={norm_out}")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved results to {OUTPUT_PATH}")
    print(f"Accuracy: {res_df['correct'].mean():.3f}")

if __name__ == "__main__":
    main()
