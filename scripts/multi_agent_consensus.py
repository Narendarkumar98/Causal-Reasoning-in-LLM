import pandas as pd
from collections import Counter
import ollama

# =====================
# CONFIG
# =====================
DATASET_PATH = r"E:\phd\exp1\Datasets\smart_building_causal_dataset_100.csv"
OUTPUT_PATH = r"E:\phd\exp1\results\multi_agent_consensus_llama3.csv"

MODEL_NAME = "llama3"
NUM_AGENTS = 3
PROMPT_CONS_PATH = r"E:\phd\exp1\prompts\consensus_agent.txt"

print("Using Ollama for multi-agent consensus:", MODEL_NAME)

with open(PROMPT_CONS_PATH, "r", encoding="utf-8") as f:
    CONSENSUS_TEMPLATE = f.read()


def call_model(prompt: str, max_new_tokens: int = 64) -> str:
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_predict": max_new_tokens},
    )
    return (resp.get("message", {}) or {}).get("content", "").strip()


def extract_final_answer(text: str) -> str:
    lower = text.lower()
    if "final answer:" in lower:
        idx = lower.rfind("final answer:")
        ans = text[idx + len("final answer:") :].strip()
        return ans.splitlines()[0].strip()
    return text.strip()


def normalize(ans: str) -> str:
    a = ans.strip()
    al = a.lower()
    if "yes" in al:
        return "Yes"
    if "no" in al:
        return "No"
    return a


def read_dataset(path: str) -> pd.DataFrame:
    """
    Robust loader:
    1) normal pd.read_csv
    2) if it loads as a SINGLE column like 'question,answer,level,scenario',
       then split that column into real columns by comma.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Case: entire header became one column
    if len(df.columns) == 1 and "question" in df.columns[0].lower() and "," in df.columns[0]:
        only_col = df.columns[0]
        print(f"Detected single-column CSV. Splitting column '{only_col}' by commas...")

        # Split header
        header_parts = [h.strip() for h in only_col.split(",")]

        # Split each row (works if your data doesn't contain commas inside fields)
        split_rows = df[only_col].astype(str).str.split(",", n=len(header_parts) - 1, expand=True)
        split_rows.columns = header_parts

        df = split_rows

    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded dataset | Columns: {list(df.columns)} | Rows: {len(df)}")
    return df


def pick_col(df: pd.DataFrame, preferred: str, fallbacks: list[str]) -> str:
    norm_map = {c.strip().lower(): c for c in df.columns}
    for name in [preferred] + fallbacks:
        key = name.strip().lower()
        if key in norm_map:
            return norm_map[key]
    raise KeyError(
        f"Could not find a column for '{preferred}'. Available columns: {list(df.columns)}"
    )


def main():
    df = read_dataset(DATASET_PATH)

    question_col = pick_col(df, "question", ["questions", "query", "prompt", "text"])
    answer_col = pick_col(df, "answer", ["ground_truth", "label", "gt", "target"])

    level_col = None
    scenario_col = None
    try:
        level_col = pick_col(df, "level", ["difficulty", "lvl"])
    except KeyError:
        pass
    try:
        scenario_col = pick_col(df, "scenario", ["context", "setting", "case"])
    except KeyError:
        pass

    print(
        "Detected columns ->",
        f"question='{question_col}', answer='{answer_col}', level='{level_col}', scenario='{scenario_col}'"
    )

    rows = []

    for idx, row in df.iterrows():
        q = str(row[question_col]).strip()
        gt = str(row[answer_col]).strip()

        indiv_answers = []
        for _ in range(NUM_AGENTS):
            prompt = CONSENSUS_TEMPLATE.format(question=q)
            raw = call_model(prompt)
            final = extract_final_answer(raw)
            norm = normalize(final)
            indiv_answers.append(norm)

        consensus_answer = Counter(indiv_answers).most_common(1)[0][0]
        correct = (consensus_answer == gt)

        rows.append(
            {
                "question": q,
                "ground_truth": gt,
                "answers": indiv_answers,
                "consensus_answer": consensus_answer,
                "correct": int(correct),
                "level": row[level_col] if level_col else "NA",
                "scenario": row[scenario_col] if scenario_col else "NA",
            }
        )

        print(
            f"[{idx+1}/{len(df)}] Answers={indiv_answers} | Consensus={consensus_answer} | GT={gt} | Correct={correct}"
        )

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved results to {OUTPUT_PATH}")
    print(f"Consensus accuracy: {res_df['correct'].mean():.3f}")


if __name__ == "__main__":
    main()


