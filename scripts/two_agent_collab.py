import pandas as pd
import ollama

# =====================
# CONFIG
# =====================
DATASET_PATH = r"E:\phd\exp1\Datasets\smart_building_causal_dataset_100.csv"
OUTPUT_PATH = r"E:\phd\exp1\results\two_agent_collab_llama3.csv"

MODEL_NAME = "llama3"

PROMPT_A_PATH = r"E:\phd\exp1\prompts\collab_agent_A.txt"
PROMPT_B_PATH = r"E:\phd\exp1\prompts\collab_agent_B.txt"

print("Using Ollama for Two-Agent Collaboration:", MODEL_NAME)

with open(PROMPT_A_PATH, "r", encoding="utf-8") as f:
    TEMPLATE_A = f.read()

with open(PROMPT_B_PATH, "r", encoding="utf-8") as f:
    TEMPLATE_B = f.read()


def call_model(prompt: str, max_new_tokens: int = 64) -> str:
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.0,
            "num_predict": max_new_tokens,
        },
    )
    return (resp.get("message", {}) or {}).get("content", "").strip()


def extract_final_answer(text: str) -> str:
    lower = text.lower()
    if "final answer:" in lower:
        idx = lower.rfind("final answer:")
        ans = text[idx + len("final answer:") :].strip()
        return ans.splitlines()[0].strip()
    return text.strip()


def normalize_answer(ans: str) -> str:
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
    - reads with utf-8-sig (handles BOM)
    - if header got read as one column 'question,answer,level,scenario',
      split that single column into real columns by comma.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")

    if len(df.columns) == 1 and "," in df.columns[0] and "question" in df.columns[0].lower():
        only_col = df.columns[0]
        print(f"Detected single-column CSV. Splitting column '{only_col}' by commas...")

        header_parts = [h.strip() for h in only_col.split(",")]
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

        # Agent A
        prompt_A = TEMPLATE_A.format(question=q)
        outA = call_model(prompt_A)

        # Agent B (sees Agent A output)
        # Keep your template variable name: agentA_message
        prompt_B = TEMPLATE_B.format(question=q, agentA_message=outA)
        outB = call_model(prompt_B)

        final_raw = extract_final_answer(outB)
        final_norm = normalize_answer(final_raw)

        correct = (final_norm == gt)

        rows.append(
            {
                "question": q,
                "ground_truth": gt,
                "agentA_output": outA,
                "agentB_output": outB,
                "final_answer": final_norm,
                "correct": int(correct),
                "level": row[level_col] if level_col else "NA",
                "scenario": row[scenario_col] if scenario_col else "NA",
            }
        )

        print(f"[{idx+1}/{len(df)}] Correct={correct} | GT={gt} | OUT={final_norm}")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved results to {OUTPUT_PATH}")
    print(f"Collab accuracy: {res_df['correct'].mean():.3f}")


if __name__ == "__main__":
    main()


