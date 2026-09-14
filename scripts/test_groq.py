#!/usr/bin/env python3
"""CLI to validate Groq API access and see which models your key can use.

Usage:
  python scripts/test_groq.py --prompt "hello"          # tries the whole fallback chain
  python scripts/test_groq.py --model llama-3.1-8b-instant --prompt "hello"

Run this whenever the Streamlit app reports a "model not found" or
connection error - it isolates whether the problem is your API key, your
network, or a specific decommissioned model, without going through the rest
of the RAG pipeline.
"""
import os
import sys
import argparse
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Test Groq API connectivity and model availability")
    parser.add_argument("--prompt", default="Say hello in one sentence.", help="User prompt to send to the model")
    parser.add_argument("--model", default=None, help="Test only this model id (default: try the whole fallback chain)")
    args = parser.parse_args()

    load_dotenv_and_check_key()

    try:
        from groq import Groq, APIStatusError, APIConnectionError
    except Exception:
        print("Python package 'groq' is not installed. Install it with:\n  pip install groq")
        sys.exit(3)

    try:
        from agents.text_to_sql import _model_chain
        models = [args.model] if args.model else _model_chain()
    except Exception:
        # Fall back to a hardcoded list if the app package can't be imported
        # (e.g. running this script from outside the project root).
        models = [args.model] if args.model else [
            "openai/gpt-oss-120b", "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "groq/compound",
        ]

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    any_ok = False

    for model in models:
        print(f"\n=== {model} ===")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a concise assistant used for connectivity tests."},
                    {"role": "user", "content": args.prompt},
                ],
            )
            content = response.choices[0].message.content
            print("OK. Model response:", content)
            any_ok = True
        except APIStatusError as e:
            print(f"FAILED (status error - likely invalid/decommissioned model or bad request): {e}")
        except APIConnectionError as e:
            print(f"FAILED (connection error - check network/proxy): {e}")
        except Exception:
            print("FAILED (unexpected error):")
            traceback.print_exc()

    print()
    if any_ok:
        print("At least one model works. Set GROQ_MODEL in your .env to pin a specific one if you like.")
        sys.exit(0)
    else:
        print("No model in the chain worked with this API key. Check your GROQ_API_KEY, network access, "
              "and https://console.groq.com/docs/models for currently supported model ids.")
        sys.exit(4)


def load_dotenv_and_check_key():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY environment variable is not set. Add it to a .env file or export it, e.g.:")
        print("  export GROQ_API_KEY=your_key_here")
        sys.exit(2)


if __name__ == "__main__":
    main()
