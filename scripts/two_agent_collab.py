import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =====================
# CONFIG
# =====================
DATASET_PATH = r"C:\Users\HP\Desktop\PhD\workshop-1\Datasets\smart_building_causal_dataset.csv"
OUTPUT_PATH = r"C:\Users\HP\Desktop\PhD\workshop-1\results\two_agent_collab_llama3.csv"
MODEL_NAME_A = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME_B = "meta-llama/Meta-Llama-3-8B-Instruct"

PROMPT_A_PATH = r"C:\Users\HP\Desktop\PhD\workshop-1\prompts\collab_agent_A.txt"
PROMPT_B_PATH = r"C:\Users\HP\Desktop\PhD\workshop-1\prompts\collab_agent_B.txt"

print("Loading Llama-3-8B-Instruct for Agent A and Agent B...")
tokenizerA = AutoTokenizer.from_pretrained(MODEL_NAME_A)
modelA = AutoModelForCausalLM.from_pretrained(MODEL_NAME_A, device_map="auto")
genA = pipeline("text-generation", model=modelA, tokenizer=tokenizerA)

tokenizerB = AutoTokenizer.from_pretrained(MODEL_NAME_B)
modelB = AutoModelForCausalLM.from_pretrained(MODEL_NAME_B, device_map="auto")
genB = pipeline("text-generation", model=modelB, tokenizer=tokenizerB)

with open(PROMPT_A_PATH, "r", encoding="utf-8") as f:
    TEMPLATE_A = f.read()

with open(PROMPT_B_PATH, "r", encoding="utf-8") as f:
    TEMPLATE_B = f.read()

def call_model(gen, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
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

def normalize_answer(ans: str) -> str:
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

        # Agent A
        prompt_A = TEMPLATE_A.format(question=q)
        outA = call_model(genA, tokenizerA, prompt_A)

        # Agent B
        prompt_B = TEMPLATE_B.format(question=q, agentA_message=outA)
        outB = call_model(genB, tokenizerB, prompt_B)
        final_raw = extract_final_answer(outB)
        final_norm = normalize_answer(final_raw)

        correct = (final_norm == gt)

        rows.append({
            "question": q,
            "ground_truth": gt,
            "agentA_output": outA,
            "agentB_output": outB,
            "final_answer": final_norm,
            "correct": int(correct),
            "level": row["level"],
            "scenario": row["scenario"],
        })

        print(f"[{idx+1}/{len(df)}] Correct: {correct} | GT={gt} | OUT={final_norm}")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved results to {OUTPUT_PATH}")
    print(f"Collab accuracy: {res_df['correct'].mean():.3f}")

if __name__ == "__main__":
    main()
