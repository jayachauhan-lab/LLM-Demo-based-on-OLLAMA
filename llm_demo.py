"""
CLI-only Ollama demo — strict LLM mode with robust ollama binary check and verification.
Dataset is loaded from an external CSV file (products_office_supplies.csv).

Behavior:
- Uses Ollama CLI (OLLAMA_BIN) to run MODEL_NAME.
- Probes the CLI with multiple commands to confirm usability.
- Pulls the configured model if missing.
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
    "Use only the data provided below; do not hallucinate. "
    "If the answer requires calculations use simple arithmetic.\n\n"
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


def run_subprocess(cmd: List[str], input_text: str = None, timeout: int = 300) -> Dict[str, str]:
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
        if res.get("ok") or (cmd[1] == "list" and res.get("stdout")):
            return True
    return False


def ensure_ollama_binary() -> None:
    if not ollama_binary_usable():
        raise RuntimeError(
            f"Ollama binary not found or not usable. "
            f"Checked '--version', 'version', 'list', 'help'. "
            f"Path tried: '{OLLAMA_BIN}'."
        )


def model_available_locally(model: str) -> bool:
    res = run_subprocess([OLLAMA_BIN, "list"], timeout=20)
    stdout = res.get("stdout", "")
    return model in stdout


def pull_model_or_raise(model: str) -> None:
    print(f"Attempting to pull model '{model}' via: {OLLAMA_BIN} pull {model}")
    res = run_subprocess([OLLAMA_BIN, "pull", model], timeout=600)
    if not res.get("ok"):
        raise RuntimeError(f"Failed to pull model '{model}'. stderr: {clean_ansi(res.get('stderr',''))}")
    print(f"Model '{model}' pulled successfully.")


def run_model_once_or_raise(model: str, prompt: str) -> str:
    res = run_subprocess([OLLAMA_BIN, "run", model], input_text=prompt, timeout=300)
    if not res.get("ok"):
        raise RuntimeError(f"Ollama run failed (rc={res.get('rc')}): {clean_ansi(res.get('stderr',''))}")
    return res.get("stdout", "").strip()


# --------- Demo wrappers ---------
def prepare_llm_or_exit(model: str) -> None:
    ensure_ollama_binary()
    if not model_available_locally(model):
        pull_model_or_raise(model)
        if not model_available_locally(model):
            raise RuntimeError(f"Model '{model}' still not available after pulling.")


def demo_product_qa_with_llm(question: str, dataset_text: str) -> str:
    prompt = PROMPT_PRODUCT_QA.format(dataset=dataset_text, question=question)
    return run_model_once_or_raise(MODEL_NAME, prompt)


# --------- Deterministic expected helpers ---------
def expected_highest_price_summary(products: List[Dict]) -> str:
    p = max(products, key=lambda x: x["price"])
    return f"{p['name']} — id: {p['id']}; price: {p['price']}"


def expected_product_details_by_id(products: List[Dict], product_id: str) -> Dict:
    for p in products:
        if p["id"] == product_id:
            return p
    raise KeyError(f"Product id {product_id} not found")


# --------- Main Examples ---------
def run_examples():
    products = load_products()
    dataset_text = products_table(products)

    prepare_llm_or_exit(MODEL_NAME)
    print(f"RUN MODE: LLM (ollama run) — MODEL={MODEL_NAME}\n")

    # ---------------- Example 1 ----------------
    print("=== Example 1: Product Q&A (LLM) ===")
    q1 = "Which product has the highest price, and what is the name and id?"
    print("Input Prompt:", q1)
    out1 = demo_product_qa_with_llm(q1, dataset_text)
    print("Output (LLM):", out1)

    expected1 = expected_highest_price_summary(products)
    print("Expected:", expected1)
    print()

    # Follow-up 1
    p1 = max(products, key=lambda x: x["stock"])
    reorder_qty1 = max(0, 1000 - p1["stock"])
    reorder_cost1 = reorder_qty1 * p1["price"]
    expected_followup1 = f"id: {p1['id']}; reorder_qty={reorder_qty1}; reorder_cost=${reorder_cost1:.2f}"

    follow1 = (
        "For the product you identified as having the highest stock, "
        "how many units are needed to reach 1000 units, and what is the reorder cost? "
        "Provide product id, reorder_qty, and reorder_cost."
    )
    print("Following Prompt:", follow1)
    out1_f = run_model_once_or_raise(MODEL_NAME, PROMPT_PRODUCT_QA.format(dataset=dataset_text, question=follow1))
    print("Output (LLM Follow-up):", out1_f)
    print("Expected Follow-up:", expected_followup1)
    print()

    # ---------------- Example 2 ----------------
    print("=== Example 2: Product detail query (LLM) ===")
    product_id = "p2"
    q2 = (
        f"Provide a concise summary of product id {product_id}: "
        f"include id, name, price, and stock."
    )
    print("Input Prompt:", q2)
    out2 = run_model_once_or_raise(MODEL_NAME, PROMPT_PRODUCT_QA.format(dataset=dataset_text, question=q2))
    print("Output (LLM):", out2)

    p2 = expected_product_details_by_id(products, product_id)
    print("Expected:", json.dumps(p2, indent=2))
    print()

    # -------- NEW UNIQUE FOLLOW-UP FOR EXAMPLE 2 --------
    avg_stock = sum(p["stock"] for p in products) / len(products)
    inventory_value = p2["price"] * p2["stock"]
    stock_position = "above" if p2["stock"] > avg_stock else "below"

    expected_followup2 = (
        f"id: {p2['id']}; inventory_value=${inventory_value:.2f}; "
        f"stock_position={stock_position}"
    )

    follow2 = (
        f"For product id {product_id}, compute the total inventory value "
        f"(price multiplied by stock). Then determine whether this product’s stock "
        f"is above or below the average stock across the entire dataset. "
        f"Provide product id, inventory_value, and stock_position."
    )

    print("Following Prompt:", follow2)
    out2_f = run_model_once_or_raise(MODEL_NAME, PROMPT_PRODUCT_QA.format(dataset=dataset_text, question=follow2))
    print("Output (LLM Follow-up):", out2_f)
    print("Expected Follow-up:", expected_followup2)
    print()


     # ---------------- Example 3 (based on Example C) ----------------
    print("=== Example 3: Product detail query (LLM) ===")
    product_id = "p17"
    q3 = (
        f"Provide a concise summary of product id {product_id}: "
        f"include id, name, price, and stock."
    )
    print("Input Prompt:", q3)
    out3 = run_model_once_or_raise(MODEL_NAME, PROMPT_PRODUCT_QA.format(dataset=dataset_text, question=q3))
    print("Output (LLM):", out3)

    p3 = expected_product_details_by_id(products, product_id)
    print("Expected:", json.dumps(p3, indent=2))
    print()

    # -------- FOLLOW-UP for Example 3 --------
    avg_stock = sum(p["stock"] for p in products) / len(products)
    stock_position = "above" if p3["stock"] > avg_stock else "below"

    expected_followup3 = f"id: {p3['id']}; stock_position={stock_position}"

    follow3 = (
        f"For product id {product_id}, determine whether its stock "
        f"is above or below the average stock across the entire dataset. "
        f"Provide product id and stock_position."
    )

    print("Following Prompt:", follow3)
    out3_f = run_model_once_or_raise(MODEL_NAME, PROMPT_PRODUCT_QA.format(dataset=dataset_text, question=follow3))
    print("Output (LLM Follow-up):", out3_f)
    print("Expected Follow-up:", expected_followup3)
    print()
  
    
# Entry point
if __name__ == "__main__":
    try:
        run_examples()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
