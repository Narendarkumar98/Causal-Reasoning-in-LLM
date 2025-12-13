import pandas as pd
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =====================
# CONFIG
# =====================
DATASET_PATH = r"C:\Users\HP\Desktop\PhD\workshop-1\Datasets\smart_building_causal_dataset.csv"
OUTPUT_PATH = r"C:\Users\HP\Desktop\PhD\workshop-1\results\multi_agent_consensus_llama3.csv"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
NUM_AGENTS = 3  # how many independent agents vote

PROMPT_CONS_PATH = r"C:\Users\HP\Desktop\PhD\workshop-1\prompts\consensus_agent.txt"

print("Loading Llama-3-8B-Instruct for multi-agent consensus...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

with open(PROMPT_CONS_PATH, "r", encoding="utf-8") as f:
    CONSENSUS_TEMPLATE = f.read()

def call_model(prompt: str, max_new_tokens: int = 128) -> str:
    out = gen(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = out[0]["generated_text"]
    return text[len(prompt):].strip()

def extract_final_answer(text: str) -> str:
    lower = text.lower()
    if "final answer:" in lower:
        idx = lower.rfind("final answer:")
        ans = text[idx + len("final answer:"):].strip()
        return ans.splitlines()[0].strip()
    return text.strip()

def normalize(ans: str) -> str:
    a = ans.strip()
    if "yes" in a.lower():
        return "Yes"
    if "no" in a.lower():
        return "No"
    return a

def main():
    df = pd.read_csv(DATASET_PATH)
    rows = []

    for idx, row in df.iterrows():
        q = row["question"]
        gt = str(row["answer"]).strip()

        indiv_answers = []

        for i in range(NUM_AGENTS):
            prompt = CONSENSUS_TEMPLATE.format(question=q)
            raw = call_model(prompt)
            final = extract_final_answer(raw)
            norm = normalize(final)
            indiv_answers.append(norm)

        counts = Counter(indiv_answers)
        consensus_answer, _ = counts.most_common(1)[0]
        correct = (consensus_answer == gt)

        rows.append({
            "question": q,
            "ground_truth": gt,
            "answers": indiv_answers,
            "consensus_answer": consensus_answer,
            "correct": int(correct),
            "level": row["level"],
            "scenario": row["scenario"],
        })

        print(f"[{idx+1}/{len(df)}] Answers={indiv_answers} | Consensus={consensus_answer} | GT={gt} | Correct={correct}")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved results to {OUTPUT_PATH}")
    print(f"Consensus accuracy: {res_df['correct'].mean():.3f}")

if __name__ == "__main__":
    main()
