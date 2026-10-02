import time

from openai import OpenAI
from transformers import AutoTokenizer

model = "HuggingFaceTB/SmolLM2-360M-Instruct"
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)
tokenizer = AutoTokenizer.from_pretrained(model)
prompt = "explain python list in exactly five short points"
max_token = 100


def run_benchmark():
    start_time = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_token,
        temperature=0,
        stream=True,
    )
    first_token_time = None
    generated_text = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
                ttft = first_token_time - start_time
                print(f"\n\nTTFT:{ttft:.3f} sec\n")
            generated_text += content
            print(content, end="", flush=True)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    output_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))

    if first_token_time is not None:
        generation_time = end_time - first_token_time
        tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0.0
        print(f"\nTotal time: {total_time:.2f} sec")
        print(f"Output tokens: {output_tokens}")
        print(f"Generation speed: {tokens_per_second:.2f} tokens/sec")


if __name__ == "__main__":
    run_benchmark()
