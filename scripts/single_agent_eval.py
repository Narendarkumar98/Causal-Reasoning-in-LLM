import pandas as pd
import ollama

# =====================
# CONFIG
# =====================
DATASET_PATH = r"E:\phd\exp1\Datasets\smart_building_causal_dataset_100.csv"
OUTPUT_PATH = r"E:\phd\exp1\results\single_agent_results_llama3.csv"

MODEL_NAME = "llama3"

SINGLE_AGENT_PROMPT = """You are an AI assistant inside a smart building.
Answer the following causal question with a single word: "Yes" or "No",
unless the question explicitly asks for a short textual answer.

Question:
{question}

Answer:
"""

print("Using Ollama for single-agent evaluation:", MODEL_NAME)


def call_model(prompt: str, max_new_tokens: int = 32) -> str:
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.0,
            "num_predict": max_new_tokens,
        },
    )
    return (resp.get("message", {}) or {}).get("content", "").strip()


def normalize_answer(ans: str) -> str:
    ans = ans.strip()
    al = ans.lower()
    if "yes" in al:
        return "Yes"
    if "no" in al:
        return "No"
    return ans


def read_dataset(path: str) -> pd.DataFrame:
    """
    Robust loader:
    - reads with utf-8-sig (handles BOM)
    - if header got read as one column 'question,answer,level,scenario',
      split that single column into real columns by comma.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Single-column CSV case (your earlier error)
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

        prompt = SINGLE_AGENT_PROMPT.format(question=q)
        model_out = call_model(prompt)
        norm_out = normalize_answer(model_out)

        correct = (norm_out == gt)

        rows.append(
            {
                "question": q,
                "ground_truth": gt,
                "raw_output": model_out,
                "normalized_output": norm_out,
                "correct": int(correct),
                "level": row[level_col] if level_col else "NA",
                "scenario": row[scenario_col] if scenario_col else "NA",
            }
        )

        print(f"[{idx+1}/{len(df)}] Correct={correct} | GT={gt} | OUT={norm_out}")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved results to {OUTPUT_PATH}")
    print(f"Accuracy: {res_df['correct'].mean():.3f}")


if __name__ == "__main__":
    main()


