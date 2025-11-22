"""
CLI-only Ollama demo — strict LLM mode with robust ollama binary check and verification.

Dataset is now loaded from an external CSV file (products.csv) instead of being
embedded directly in the code.

Behavior:
- Uses Ollama CLI (OLLAMA_BIN) to run MODEL_NAME.
- Probes the CLI with multiple commands to confirm usability.
- Pulls the configured model if missing.
- No fallback; script exits on failures.
- Two LLM-driven examples with verification + follow-ups.
"""

import os
import csv
import json
import re
import subprocess
import sys
from typing import List, Dict

# ----------------- Configuration -----------------
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BIN = os.getenv("OLLAMA_BIN", r"C:\Users\chauh\AppData\Local\Programs\Ollama\ollama.exe")
DATASET_FILE = "products.csv"
ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
# -------------------------------------------------


# --------- Load CSV dataset ---------
def load_products() -> List[Dict]:
    products = []
    with open(DATASET_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["price"] = float(row["price"])
            row["stock"] = int(row["stock"])
            products.append(row)
    return products


# --------- Prompt templates ---------
PROMPT_PRODUCT_QA = (
    "You are a helpful assistant that answers questions about a product dataset.\n"
    "Use only the data provided below; do not hallucinate. If the answer requires calculations use simple arithmetic.\n\n"
    "Dataset:\n{dataset}\n\n"
    "Question: {question}\n\n"
    "Provide a concise answer and, when helpful, include the id of the referenced product."
)


# --------- Helpers ---------
def products_table(products: List[Dict]) -> str:
    lines = ["id | name | price | stock", "---|---|---:|---"]
    for p in products:
        lines.append(f"{p['id']} | {p['name']} | {p['price']} | {p['stock']}")
    return "\n".join(lines)


def clean_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def run_subprocess(cmd: List[str], input_text: str = None, timeout: int = 60) -> Dict[str, str]:
    try:
        proc = subprocess.run(
            cmd,
            input=input_text.encode("utf-8") if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {"ok": False, "stderr": f"'{cmd[0]}' not found", "rc": None}
    except subprocess.TimeoutExpired:
        return {"ok": False, "stderr": "process timed out", "rc": None}

    stdout = proc.stdout.decode(errors="ignore")
    stderr = proc.stderr.decode(errors="ignore")
    return {"ok": proc.returncode == 0, "stdout": stdout, "stderr": stderr, "rc": proc.returncode}


def ollama_binary_usable() -> bool:
    probes = [
        [OLLAMA_BIN, "--version"],
        [OLLAMA_BIN, "version"],
        [OLLAMA_BIN, "list"],
        [OLLAMA_BIN, "help"],
    ]
    for cmd in probes:
        try:
            res = run_subprocess(cmd, timeout=8)
        except Exception:
            continue
        if res.get("ok"):
            return True
        if cmd[1] == "list" and res.get("stdout"):
            return True
    return False


def ensure_ollama_binary() -> None:
    if not ollama_binary_usable():
        raise RuntimeError(
            f"ollama binary not found or not usable. Probed '--version', 'version', 'list', 'help'. "
            f"Ensure '{OLLAMA_BIN}' is correct and runnable."
        )


def model_available_locally(model: str) -> bool:
    res = run_subprocess([OLLAMA_BIN, "list"], timeout=1000)
    stdout = res.get("stdout", "")
    return model in stdout


def pull_model_or_raise(model: str) -> None:
    print(f"Attempting to pull model '{model}' via: {OLLAMA_BIN} pull {model}")
    res = run_subprocess([OLLAMA_BIN, "pull", model], timeout=600)
    if not res.get("ok"):
        raise RuntimeError(f"Failed to pull model '{model}'. stderr: {clean_ansi(res.get('stderr',''))}")
    print(f"Model '{model}' pulled successfully.")


def run_model_once_or_raise(model: str, prompt: str) -> str:
    res = run_subprocess([OLLAMA_BIN, "run", model], input_text=prompt, timeout=120)
    if not res.get("ok"):
        raise RuntimeError(f"ollama run failed: {clean_ansi(res.get('stderr',''))}")
    return res.get("stdout", "").strip()


# --------- Demo API wrappers (strict LLM mode) ---------
def prepare_llm_or_exit(model: str) -> None:
    ensure_ollama_binary()
    if model_available_locally(model):
        return
    pull_model_or_raise(model)
    if not model_available_locally(model):
        raise RuntimeError(f"Model '{model}' not found after pull attempt.")


def demo_product_qa_with_llm(question: str, dataset_text: str) -> str:
    prompt = PROMPT_PRODUCT_QA.format(dataset=dataset_text, question=question)
    return run_model_once_or_raise(MODEL_NAME, prompt)


# --------- Deterministic expected-answer helpers ---------
def expected_highest_price_summary(products: List[Dict]) -> str:
    p = max(products, key=lambda x: x["price"])
    return f"{p['name']} — id: {p['id']}; price: {p['price']}"


def expected_product_details_by_id(products: List[Dict], product_id: str) -> Dict:
    for p in products:
        if p["id"] == product_id:
            return p
    raise KeyError(f"product id {product_id} not found")


# --------- Example runs ---------
def run_examples():
    # Load external CSV
    products = load_products()
    dataset_text = products_table(products)

    prepare_llm_or_exit(MODEL_NAME)
    print(f"RUN MODE: LLM (ollama run)  — OLLAMA_BIN={OLLAMA_BIN} MODEL={MODEL_NAME}\n")

    # ---------------- Example 1 ----------------
    print("=== Example 1: Product Q&A (LLM) ===")
    input1 = "Which product has the highest price, and what is the name and id?"
    print("Input Prompt:", input1)
    output1 = demo_product_qa_with_llm(input1, dataset_text)
    print("Output (LLM):", output1)

    expected1 = expected_highest_price_summary(products)
    print("Expected (from CSV):", expected1)
    if expected1 not in output1:
        print("WARNING: LLM output does not match expected.", file=sys.stderr)
    print()

    # -------- Example 1 Follow-up --------
    p1 = max(products, key=lambda x: x["stock"])
    reorder_qty1 = max(0, 1000 - p1["stock"])
    reorder_cost1 = reorder_qty1 * p1["price"]
    expected_followup1 = f"id: {p1['id']}; reorder_qty={reorder_qty1}; reorder_cost=${reorder_cost1:.2f}"

    following_prompt1 = (
        "For the product you identified as having the highest stock, how many units are needed to reach 1000 units, "
        "and what is the reorder cost? Show the product id, reorder_qty and reorder_cost."
    )

    print("Following Prompt:", following_prompt1)
    combined_followup1 = PROMPT_PRODUCT_QA.format(dataset=dataset_text, question=following_prompt1)
    output1_followup = run_model_once_or_raise(MODEL_NAME, combined_followup1)
    print("Output (LLM Follow-up):", output1_followup)
    print("Expected Follow-up:", expected_followup1)
    print()

    # ---------------- Example 2 ----------------
    print("=== Example 2: Product detail query (LLM) ===")
    product_id = "p2"
    input2 = f"Provide a concise summary of product id {product_id}: include id, name, price, and stock."
    print("Input Prompt:", input2)

    dataset_prompt = PROMPT_PRODUCT_QA.format(dataset=dataset_text, question=input2)
    output2 = run_model_once_or_raise(MODEL_NAME, dataset_prompt)
    print("Output (LLM):", output2)

    expected2 = expected_product_details_by_id(products, product_id)
    print("Expected:", json.dumps(expected2, indent=2))
    print()

    # -------- Example 2 Follow-up --------
    p2 = expected_product_details_by_id(products, product_id)
    reorder_qty2 = max(0, 1000 - p2["stock"])
    reorder_cost2 = reorder_qty2 * p2["price"]
    expected_followup2 = f"id: {p2['id']}; reorder_qty={reorder_qty2}; reorder_cost=${reorder_cost2:.2f}"

    following_prompt2 = (
        f"For product id {product_id}, how many units are needed to reach 1000 units, "
        f"and what is the reorder cost? Provide product id, reorder_qty, and reorder_cost."
    )

    print("Following Prompt:", following_prompt2)
    combined_followup2 = PROMPT_PRODUCT_QA.format(dataset=dataset_text, question=following_prompt2)
    output2_followup = run_model_once_or_raise(MODEL_NAME, combined_followup2)
    print("Output (LLM Follow-up):", output2_followup)
    print("Expected Follow-up:", expected_followup2)
    print()


# Entry point
if __name__ == "__main__":
    try:
        run_examples()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
