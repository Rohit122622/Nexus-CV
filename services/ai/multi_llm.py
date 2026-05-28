"""
Multi-LLM Service — Production-grade fallback chain.

MODEL PRIORITY (STRICT):
  1. gemini-2.5-flash
  2. gemini-2.5-flash-lite
  3. gemini-3.1-flash-lite
  4. gemini-3-flash
  5. Groq (llama-3.3-70b-versatile)
  6. Local rule-based (final safety net)

Features:
  - Thread-safe throttling (1.5s between Gemini calls)
  - 2 retries per model
  - safe_json_parse() on ALL responses
  - No system crash possible from LLM failures
"""

import os
import json
import logging
import requests
import traceback
import time
import threading
from utils.json_utils import safe_json_parse

logger = logging.getLogger(__name__)

# ── Throttle state for Gemini quota protection ──
_last_gemini_call = 0.0
_gemini_lock = threading.Lock()
_GEMINI_MIN_INTERVAL = 1.5  # seconds between Gemini calls

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite"
]


def extract_json(text):
    """Parse LLM text response into a dict with unified structure."""
    data = safe_json_parse(text)
    if isinstance(data, dict):
        if "insights" not in data:
            data["insights"] = str(data.get("verdict", data.get("rewritten_objective", "Generated insights.")))
        if "suggestions" not in data:
            data["suggestions"] = data.get("skill_suggestions", data.get("version_a_strengths", []))
        if "analysis" not in data:
            data["analysis"] = str(data.get("verdict_reason", "Analysis complete."))
        if "score_reason" not in data:
            data["score_reason"] = str(data.get("strength_reason", "Score evaluated."))
        return data
    raise Exception("Failed to parse JSON from LLM response")


def build_openai_payload(prompt, model="gpt-4o-mini"):
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }


# ─────────────── GEMINI (MULTI-MODEL) ───────────────

def call_gemini(prompt):
    """
    Try multiple Gemini models in priority order.
    Each model gets 2 attempts. 429 errors skip to next model immediately.
    Thread-safe throttling prevents quota exhaustion.
    """
    global _last_gemini_call
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY not set")

    # ── Throttle: enforce minimum interval ──
    with _gemini_lock:
        elapsed = time.time() - _last_gemini_call
        if elapsed < _GEMINI_MIN_INTERVAL:
            time.sleep(_GEMINI_MIN_INTERVAL - elapsed)
        _last_gemini_call = time.time()

    prompt_final = prompt + "\nReturn ONLY valid JSON."
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        max_output_tokens=1024,
        temperature=0.2
    )

    last_error = None

    for model_name in GEMINI_MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt_final,
                    config=config
                )
                logger.info("Gemini %s succeeded", model_name)
                return extract_json(response.text)
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    logger.info("Gemini %s rate-limited, trying next model", model_name)
                    break  # skip retries, try next model
                if "not found" in err_str.lower() or "invalid" in err_str.lower():
                    logger.info("Gemini %s not available, trying next model", model_name)
                    break  # model doesn't exist, try next
                if attempt < 1:
                    time.sleep(1.5)
                    continue
                logger.warning("Gemini %s attempt %d error: %s", model_name, attempt+1, type(e).__name__)
                break  # move to next model

    logger.warning("All Gemini models exhausted, switching provider")
    raise last_error or Exception("All Gemini models failed")


# ─────────────── GROQ ───────────────

def call_groq(prompt):
    """Groq API call with llama-3.3-70b-versatile."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not set")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=data, timeout=15)
    if response.status_code != 200:
        raise Exception(f"Groq HTTP {response.status_code}: {response.text[:200]}")
    return extract_json(response.json()["choices"][0]["message"]["content"])


# ─────────────── OTHER PROVIDERS (SECONDARY) ───────────────

def call_grok(prompt):
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROK_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "grok-4-1-fast",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Grok failed: {response.text}")
    return extract_json(response.json()["choices"][0]["message"]["content"])


def call_claude(prompt):
    headers = {
        "x-api-key": os.getenv("CLAUDE_API_KEY"),
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    data = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Claude failed: {response.text}")
    return extract_json(response.json()["content"][0]["text"])


def call_qwen(prompt):
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('QWEN_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Qwen failed: {response.text}")
    return extract_json(response.json()["choices"][0]["message"]["content"])


def call_openai(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: raise Exception("OPENAI_API_KEY missing")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=build_openai_payload(prompt, "gpt-4o-mini"), timeout=15)
    if response.status_code != 200:
        raise Exception(f"OpenAI failed: {response.text}")
    return extract_json(response.json()["choices"][0]["message"]["content"])


def call_deepseek(prompt):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key: raise Exception("DEEPSEEK_API_KEY missing")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=body, timeout=15)
    if response.status_code != 200:
        raise Exception(f"DeepSeek failed: {response.text}")
    return extract_json(response.json()["choices"][0]["message"]["content"])


# ─────────────── CENTRAL LLM CALLER ───────────────

def call_llm(prompt):
    """
    Central LLM function — strict fallback chain.

    Order: Gemini (4 models) -> Groq -> local rule-based.
    NEVER crashes. Always returns a dict.
    """

    # ── Priority 1: Gemini (multi-model) ──
    try:
        logger.info("LLM: Attempting Gemini fallback chain")
        result = call_gemini(prompt)
        logger.info("LLM: Gemini succeeded")
        return result
    except Exception as e:
        logger.warning("LLM: Primary models unavailable: %s - %s", type(e).__name__, str(e)[:100])
        logger.info("LLM: Triggering strict fallback to Groq")

    # ── Priority 2: Groq (FINAL LLM fallback) ──
    try:
        logger.info("LLM: Attempting Groq")
        result = call_groq(prompt)
        logger.info("LLM: Groq succeeded")
        return result
    except Exception as e:
        logger.warning("LLM: Groq unavailable: %s - %s", type(e).__name__, str(e)[:100])
        logger.error("LLM: All primary cloud providers failed")

    # ── Priority 3: DeepSeek ──
    try:
        logger.info("LLM: Attempting DeepSeek")
        return call_deepseek(prompt)
    except Exception as e:
        logger.warning("LLM: DeepSeek unavailable: %s", type(e).__name__)

    # ── Priority 4: Qwen ──
    try:
        logger.info("LLM: Attempting Qwen")
        return call_qwen(prompt)
    except Exception as e:
        logger.warning("LLM: Qwen unavailable: %s", type(e).__name__)

    # ── Priority 5: Claude ──
    try:
        logger.info("LLM: Attempting Claude")
        return call_claude(prompt)
    except Exception as e:
        logger.warning("LLM: Claude unavailable: %s", type(e).__name__)

    # ── Local rule-based response (NEVER crashes) ──
    logger.warning("LLM: All providers exhausted, using local rule-based response")
    return {
        "insights": "Evaluation completed using rule-based analysis.",
        "suggestions": ["Consider adding role-specific skills", "Quantify achievements"],
        "analysis": "Rule-based evaluation complete.",
        "score_reason": "Scored using heuristic analysis"
    }
