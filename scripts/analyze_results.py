import os
import pandas as pd

ROOT = r"E:\phd\exp1"
RESULTS_DIR = os.path.join(ROOT, "results")

SINGLE_PATH = os.path.join(RESULTS_DIR, "single_agent_results_llama3.csv")
COLLAB_PATH = os.path.join(RESULTS_DIR, "two_agent_collab_llama3.csv")
CONS_PATH   = os.path.join(RESULTS_DIR, "multi_agent_consensus_llama3.csv")

TRUST_CSV_OUT = os.path.join(RESULTS_DIR, "trust_scores_llama3.csv")
XLSX_OUT      = os.path.join(RESULTS_DIR, "llama3_analysis.xlsx")


def norm_yesno(x):
    """Normalize any text-ish output to Yes/No if possible."""
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if "yes" in s:
        return "Yes"
    if "no" in s:
        return "No"
    return str(x).strip()


def _fix_single_column_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a CSV got read as a single column whose name contains commas
    (e.g., 'question,answer,level,scenario'), split it into real columns.
    This was the exact issue you hit with the dataset CSV; it can also happen
    with results CSVs depending on how they were generated/saved.
    """
    if len(df.columns) == 1:
        col0 = df.columns[0]
        if isinstance(col0, str) and "," in col0:
            header_parts = [h.strip() for h in col0.split(",")]
            # Split each row into columns; n limits splits to keep last column intact
            split_df = df[col0].astype(str).str.split(",", n=len(header_parts) - 1, expand=True)
            split_df.columns = header_parts
            return split_df
    return df


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file:\n  {path}\n\n"
            f"Check that your scripts produced it inside:\n  {RESULTS_DIR}"
        )

    # utf-8-sig handles BOM. This is common on Windows/Excel.
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Fix "single column" CSV edge case if it happens
    df = _fix_single_column_csv(df)

    # Strip whitespace from headers just in case
    df.columns = [str(c).strip() for c in df.columns]
    return df


def require_cols(df: pd.DataFrame, name: str, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}\nAvailable: {list(df.columns)}")


def main():
    # ---- Load
    single = safe_read_csv(SINGLE_PATH).copy()
    collab = safe_read_csv(COLLAB_PATH).copy()
    cons   = safe_read_csv(CONS_PATH).copy()

    # ---- Validate minimum columns
    require_cols(single, "Single-agent CSV", ["question", "ground_truth", "level", "scenario", "correct"])
    require_cols(collab, "Two-agent CSV",   ["question", "correct", "final_answer"])
    require_cols(cons,   "Consensus CSV",   ["question", "correct", "consensus_answer"])

    # ---- Normalize answers
    if "normalized_output" in single.columns:
        single["single_answer"] = single["normalized_output"].apply(norm_yesno)
    elif "raw_output" in single.columns:
        single["single_answer"] = single["raw_output"].apply(norm_yesno)
    else:
        raise ValueError("Single results missing normalized_output/raw_output columns")

    collab["collab_answer"] = collab["final_answer"].apply(norm_yesno)
    cons["consensus_answer_norm"] = cons["consensus_answer"].apply(norm_yesno)

    # ---- Reduce for merge
    single_small = single[
        ["question", "ground_truth", "level", "scenario", "single_answer", "correct"]
    ].rename(columns={"correct": "single_correct"})

    collab_small = collab[
        ["question", "collab_answer", "correct"]
    ].rename(columns={"correct": "collab_correct"})

    cons_small = cons[
        ["question", "consensus_answer_norm", "correct"]
    ].rename(columns={"correct": "consensus_correct"})

    merged = (
        single_small
        .merge(collab_small, on="question", how="outer")
        .merge(cons_small, on="question", how="outer")
    )

    # ---- Fill missing metadata if any (from any file that has it)
    def fill_from_any(colname: str):
        nonlocal merged
        if colname in merged.columns and merged[colname].isna().any():
            maps = []
            for df_ in (single, collab, cons):
                if colname in df_.columns:
                    maps.append(df_[["question", colname]])
            if maps:
                m = pd.concat(maps).dropna().drop_duplicates("question")
                merged2 = merged.merge(m, on="question", how="left", suffixes=("", "_fill"))
                merged2[colname] = merged2[colname].fillna(merged2[colname + "_fill"])
                merged2.drop(columns=[colname + "_fill"], inplace=True)
                merged = merged2

    fill_from_any("ground_truth")
    fill_from_any("level")
    fill_from_any("scenario")

    # ---- Trust-impact metrics
    merged["improved_over_single"] = (merged["single_correct"] == 0) & (merged["consensus_correct"] == 1)
    merged["improved_over_collab"] = (merged["collab_correct"] == 0) & (merged["consensus_correct"] == 1)
    merged["worse_than_single"]    = (merged["single_correct"] == 1) & (merged["consensus_correct"] == 0)
    merged["worse_than_collab"]    = (merged["collab_correct"] == 1) & (merged["consensus_correct"] == 0)

    merged["agree_single_collab"]    = merged["single_answer"] == merged["collab_answer"]
    merged["agree_single_consensus"] = merged["single_answer"] == merged["consensus_answer_norm"]
    merged["agree_collab_consensus"] = merged["collab_answer"] == merged["consensus_answer_norm"]

    trust_cols = [
        "question", "ground_truth", "level", "scenario",
        "single_answer", "collab_answer", "consensus_answer_norm",
        "single_correct", "collab_correct", "consensus_correct",
        "improved_over_single", "improved_over_collab",
        "worse_than_single", "worse_than_collab",
        "agree_single_collab", "agree_single_consensus", "agree_collab_consensus",
    ]

    trust = merged[trust_cols].rename(columns={"consensus_answer_norm": "consensus_answer"})
    trust.to_csv(TRUST_CSV_OUT, index=False, encoding="utf-8-sig")

    # ---- Summary tables
    def acc(col):
        s = trust[col].dropna()
        return float(s.mean()) if len(s) else 0.0

    overall_summary = pd.DataFrame([
        {"metric": "n_questions_merged", "value": int(len(trust))},
        {"metric": "single_accuracy", "value": acc("single_correct")},
        {"metric": "collab_accuracy", "value": acc("collab_correct")},
        {"metric": "consensus_accuracy", "value": acc("consensus_correct")},
        {"metric": "consensus_fixed_single_errors", "value": int(trust["improved_over_single"].sum())},
        {"metric": "consensus_fixed_collab_errors", "value": int(trust["improved_over_collab"].sum())},
        {"metric": "consensus_made_single_worse", "value": int(trust["worse_than_single"].sum())},
        {"metric": "consensus_made_collab_worse", "value": int(trust["worse_than_collab"].sum())},
    ])

    by_level = trust.groupby("level", dropna=False).agg(
        single_accuracy=("single_correct", "mean"),
        collab_accuracy=("collab_correct", "mean"),
        consensus_accuracy=("consensus_correct", "mean"),
        n=("question", "count"),
    ).reset_index()

    by_scenario = trust.groupby("scenario", dropna=False).agg(
        single_accuracy=("single_correct", "mean"),
        collab_accuracy=("collab_correct", "mean"),
        consensus_accuracy=("consensus_correct", "mean"),
        n=("question", "count"),
    ).reset_index()

    agreement = pd.DataFrame([
        {"pair": "single_vs_collab", "agreement_rate": float(trust["agree_single_collab"].mean())},
        {"pair": "single_vs_consensus", "agreement_rate": float(trust["agree_single_consensus"].mean())},
        {"pair": "collab_vs_consensus", "agreement_rate": float(trust["agree_collab_consensus"].mean())},
    ])

    # ---- Write Excel
    with pd.ExcelWriter(XLSX_OUT, engine="openpyxl") as writer:
        overall_summary.to_excel(writer, sheet_name="overall_summary", index=False)
        by_level.to_excel(writer, sheet_name="by_level", index=False)
        by_scenario.to_excel(writer, sheet_name="by_scenario", index=False)
        agreement.to_excel(writer, sheet_name="agreement", index=False)
        trust.to_excel(writer, sheet_name="raw_trust_scores", index=False)

    print("Saved:")
    print(" -", TRUST_CSV_OUT)
    print(" -", XLSX_OUT)


if __name__ == "__main__":
    main()


