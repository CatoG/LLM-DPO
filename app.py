import os
from typing import List, Dict
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F

import gradio as gr
import pandas as pd

from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer


# =========================================================
#  MODEL LIST
# =========================================================

MODEL_CHOICES = [
    # Very small / light (good for CPU Spaces)
    "distilgpt2",
    "gpt2",
    "sshleifer/tiny-gpt2",
    "LiquidAI/LFM2-350M",
    "google/gemma-3-270m-it",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "mkurman/NeuroBLAST-V3-SYNTH-EC-150000",

    # Small–medium (~1–2B) – still reasonable on CPU, just slower
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "google/gemma-3-1b-it",
    "meta-llama/Llama-3.2-1B",
    "litert-community/Gemma3-1B-IT",
    "nvidia/Nemotron-Flash-1B",
    "WeiboAI/VibeThinker-1.5B",
    "Qwen/Qwen3-1.7B",

    # Medium (~2–3B) – probably OK on beefier CPU / small GPU
    "google/gemma-2-2b-it",
    "thu-pacman/PCMind-2.1-Kaiyuan-2B",
    "opendatalab/MinerU-HTML",
    "ministral/Ministral-3b-instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "nvidia/Nemotron-Flash-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",

    # Heavier (4–8B) – you really want a GPU Space for these
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-4B-Instruct-2507",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "allenai/Olmo-3-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "openbmb/MiniCPM4.1-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "rl-research/DR-Tulu-8B",
]

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAINED_MODEL_DIR = "trained_model"


# =========================================================
#  GLOBALS & CONFIG
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
policy_model = None
ref_model = None

DEFAULT_DPO_CONFIG = DPOConfig(
    beta=0.1,
    output_dir="dpo_demo",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    remove_unused_columns=False,
    logging_steps=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    evaluation_strategy="no",
    warmup_steps=0,
    fp16=False,
    save_steps=0,
    report_to="none",
)


# =========================================================
#  LORA TARGET-MODULE HELPER
# =========================================================

def guess_lora_target_modules(model_name: str, base_model) -> List[str]:
    """
    Heuristically choose good LoRA target modules based on the model type/name.
    - GPT-2-like: use c_attn/c_proj
    - LLaMA/Gemma/Mistral/Qwen/etc: use q/k/v/o + MLP projections
    - Fallback: scan Linear module names for known patterns
    """
    model_type = getattr(base_model.config, "model_type", "") or ""
    name_lower = model_name.lower()

    # GPT-2 / DistilGPT-2 / Tiny GPT-2
    if (
        "gpt2" in model_type
        or "gpt2" in name_lower
        or "tiny-gpt2" in name_lower
        or "distilgpt2" in name_lower
    ):
        return ["c_attn", "c_proj"]

    # LLaMA / Gemma / Mistral / Qwen / Olmo / MiniCPM / SmolLM / Nemotron etc.
    if any(
        t in model_type
        for t in [
            "llama",
            "gemma",
            "mistral",
            "qwen",
            "qwen2",
            "olmo",
            "minicpm",
            "smollm",
            "nemotron",
        ]
    ):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Fallback: inspect Linear modules and see what’s there
    linear_leaf_names = []
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear):
            linear_leaf_names.append(name.split(".")[-1])

    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "c_attn", "c_proj",
    ]
    found = sorted(set(n for n in candidates if n in linear_leaf_names))
    if found:
        return found

    # If absolutely nothing matches, bail with a clear error
    raise ValueError(
        f"Could not guess LoRA target modules for model '{model_name}' "
        f"(model_type='{model_type}'). "
        f"Try setting target_modules manually for this model."
    )


# =========================================================
#  MODEL LOADING
# =========================================================

def load_base_model(model_name: str) -> str:
    """
    Load tokenizer + base model, then create:
      - policy_model: LoRA-adapted (trainable)
      - ref_model: frozen base model for DPO
    """
    global tokenizer, policy_model, ref_model

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.eos_token_id

    # Choose LoRA target modules dynamically
    target_modules = guess_lora_target_modules(model_name, base_model)

    peft_config = LoraConfig(
        r=4,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
    )

    # Policy model = base + LoRA (trainable)
    policy = get_peft_model(base_model, peft_config)
    policy.to(device)
    policy.eval()

    # Reference model = frozen base model
    reference = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    reference.config.use_cache = False
    reference.config.pad_token_id = tokenizer.eos_token_id
    reference.to(device)
    for p in reference.parameters():
        p.requires_grad = False
    reference.eval()

    policy_model = policy
    ref_model = reference

    return (
        f"Loaded base model: **{model_name}** on **{device}** "
        f"with LoRA target_modules={target_modules}"
    )


# Load default on startup
initial_status = load_base_model(DEFAULT_MODEL)


# =========================================================
#  UTILS
# =========================================================

def build_generation_config(
    do_sample: bool,
    temperature: float,
    max_new_tokens: int,
    top_k: int = 20,
    top_p: float = 0.9,
) -> GenerationConfig:
    """
    Helper to build a GenerationConfig from UI settings.
    """
    temperature = max(0.0, float(temperature))
    max_new_tokens = int(max_new_tokens)
    return GenerationConfig(
        do_sample=bool(do_sample),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )


def generate_text(
    model: nn.Module,
    prompt: str,
    gen_config: GenerationConfig,
    style_prefix: str = "",
) -> str:
    model.eval()
    full_prompt = style_prefix + prompt

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        padding=False,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=gen_config.do_sample,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            temperature=gen_config.temperature,
            max_new_tokens=gen_config.max_new_tokens,
            pad_token_id=gen_config.pad_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if text.startswith(full_prompt):
        return text[len(full_prompt):].strip()
    return text.strip()


def preferences_to_df(preferences: List[Dict]) -> pd.DataFrame:
    if not preferences:
        return pd.DataFrame(columns=["prompt", "chosen", "rejected"])
    return pd.DataFrame(preferences)


def list_trained_model_files() -> List[str]:
    """
    Return a list of filepaths under TRAINED_MODEL_DIR (for download).
    """
    if not os.path.isdir(TRAINED_MODEL_DIR):
        return []
    files: List[str] = []
    for root, dirs, filenames in os.walk(TRAINED_MODEL_DIR):
        for name in filenames:
            files.append(os.path.join(root, name))
    return files


def logprob_answer(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    answer: str,
) -> float:
    """
    Compute the log-probability of `answer` given `prompt`,
    using a simple "User/Assistant" format:

        full_text = "User: <prompt>\\nAssistant: <answer>"

    We approximate p(answer | prompt) by summing log-probs of all tokens
    in the answer region (the shared prompt part cancels in comparisons).
    """
    model.eval()
    with torch.no_grad():
        full_text = f"User: {prompt}\nAssistant: {answer}"
        enc = tokenizer(
            full_text,
            return_tensors="pt",
        ).to(device)

        input_ids = enc["input_ids"]
        out = model(input_ids=input_ids)
        logits = out.logits[:, :-1, :]          # [B, T-1, V]
        labels = input_ids[:, 1:]              # [B, T-1]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        total_logprob = token_log_probs.sum().item()

    return float(total_logprob)


# =========================================================
#  DPO CALLBACKS
# =========================================================

def generate_candidates(
    prompt: str,
    do_sample: bool,
    temperature: float,
    max_new_tokens: int,
) -> tuple[str, str]:
    """
    Generate Answer A (balanced) and Answer B (creative-ish),
    using the same core generation settings from the GUI.
    """
    if not prompt.strip():
        return "", ""

    balanced_config = build_generation_config(
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_k=20,
        top_p=0.9,
    )

    creative_temp = float(temperature) + 0.4
    creative_config = build_generation_config(
        do_sample=do_sample,
        temperature=creative_temp,
        max_new_tokens=max_new_tokens,
        top_k=50,
        top_p=0.95,
    )

    style_balanced = (
        "You are a helpful, careful assistant. "
        "Answer clearly and sensibly.\n\nUser: "
    )
    style_creative = (
        "You are a creative assistant who explores unusual ideas and stronger opinions, "
        "while still staying safe.\n\nUser: "
    )

    answer_a = generate_text(
        policy_model,
        prompt,
        balanced_config,
        style_prefix=style_balanced,
    )
    answer_b = generate_text(
        policy_model,
        prompt,
        creative_config,
        style_prefix=style_creative,
    )

    return answer_a, answer_b


def save_preference(
    prompt: str,
    answer_a: str,
    answer_b: str,
    custom_answer: str,
    preference_mode: str,
    state_preferences: List[Dict],
):
    """
    Encode a preference in one of four ways:
      - Prefer A over B        -> chosen=A, rejected=B
      - Prefer B over A        -> chosen=B, rejected=A
      - Prefer custom over A   -> chosen=custom, rejected=A
      - Prefer custom over B   -> chosen=custom, rejected=B
    """
    msg = ""

    if not prompt.strip():
        msg = "No prompt provided."
        return state_preferences, preferences_to_df(state_preferences), msg

    if not answer_a.strip() or not answer_b.strip():
        msg = "Generate both model answers before saving a preference."
        return state_preferences, preferences_to_df(state_preferences), msg

    if not preference_mode:
        msg = "Please choose how to encode the preference."
        return state_preferences, preferences_to_df(state_preferences), msg

    preference_mode = preference_mode.strip()

    chosen = None
    rejected = None

    if preference_mode == "Prefer A over B":
        chosen = answer_a
        rejected = answer_b

    elif preference_mode == "Prefer B over A":
        chosen = answer_b
        rejected = answer_a

    elif preference_mode == "Prefer custom over A":
        if not custom_answer.strip():
            msg = "You selected 'Prefer custom over A' but did not provide a custom answer."
            return state_preferences, preferences_to_df(state_preferences), msg
        chosen = custom_answer
        rejected = answer_a

    elif preference_mode == "Prefer custom over B":
        if not custom_answer.strip():
            msg = "You selected 'Prefer custom over B' but did not provide a custom answer."
            return state_preferences, preferences_to_df(state_preferences), msg
        chosen = custom_answer
        rejected = answer_b

    else:
        msg = f"Unknown preference mode: {preference_mode}"
        return state_preferences, preferences_to_df(state_preferences), msg

    entry = {
        "prompt": prompt.strip(),
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
    }

    state_preferences = list(state_preferences) + [entry]
    df = preferences_to_df(state_preferences)
    msg = f"Saved preference #{len(state_preferences)}."

    return state_preferences, df, msg


def train_dpo_model(
    state_preferences: List[Dict],
    num_epochs: int,
    learning_rate: float,
    beta: float,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Run DPO training on the accumulated preferences.
    Shows a progress bar/spinner and returns:
      - a detailed status message
      - a 'last trained' timestamp string
      - a list of saved model files for download
    """
    global policy_model, ref_model

    progress(0.0, desc="Checking preferences...")

    if not state_preferences:
        return (
            "⚠️ No preferences collected yet. Add some first.",
            "**Last trained:** never",
            [],
        )

    dataset = Dataset.from_list(state_preferences)

    progress(0.2, desc="Configuring DPO trainer...")

    dpo_config = DPOConfig(
        **{
            **DEFAULT_DPO_CONFIG.to_dict(),
            "num_train_epochs": int(num_epochs),
            "learning_rate": float(learning_rate),
            "beta": float(beta),
        }
    )

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        max_length=256,
    )

    progress(0.4, desc="Training model with DPO...")

    trainer.train()

    progress(0.75, desc="Finalizing and moving model to device...")

    policy_model = trainer.model
    policy_model.to(device)
    policy_model.eval()

    # Save the trained model + tokenizer so you can download them
    progress(0.9, desc="Saving trained model to disk...")

    os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
    policy_model.save_pretrained(TRAINED_MODEL_DIR)
    tokenizer.save_pretrained(TRAINED_MODEL_DIR)

    files = list_trained_model_files()

    progress(1.0, desc="Done")

    n = len(state_preferences)
    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    msg = f"""### ✅ Training complete

- Preference pairs used: **{n}**
- Epochs: **{num_epochs}**
- Learning rate: **{learning_rate}**
- DPO beta (strength): **{beta}**

The tuned policy model + tokenizer have been saved to `{TRAINED_MODEL_DIR}/`.
You can download them using the file list below.
"""

    last_trained_msg = f"**Last trained:** {finished_at}"

    return msg, last_trained_msg, files


def dpo_diagnostics(state_preferences: List[Dict]) -> str:
    """
    Compute how often the policy_model and ref_model
    assign higher log-probability to the CHOSEN answer
    than to the REJECTED answer.

    Returns a markdown report with:
      - number of pairs
      - policy win rate
      - ref win rate
      - average logprob margins
    """
    if not state_preferences:
        return "No preferences collected yet – nothing to evaluate."

    if policy_model is None or ref_model is None or tokenizer is None:
        return "Models not loaded – reload base model first."

    n = len(state_preferences)
    policy_wins = 0
    ref_wins = 0

    policy_margins = []
    ref_margins = []

    for ex in state_preferences:
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        # Policy model logprobs
        lp_pol_ch = logprob_answer(policy_model, tokenizer, prompt, chosen)
        lp_pol_rj = logprob_answer(policy_model, tokenizer, prompt, rejected)
        margin_pol = lp_pol_ch - lp_pol_rj
        policy_margins.append(margin_pol)
        if margin_pol > 0:
            policy_wins += 1

        # Reference model logprobs
        lp_ref_ch = logprob_answer(ref_model, tokenizer, prompt, chosen)
        lp_ref_rj = logprob_answer(ref_model, tokenizer, prompt, rejected)
        margin_ref = lp_ref_ch - lp_ref_rj
        ref_margins.append(margin_ref)
        if margin_ref > 0:
            ref_wins += 1

    policy_winrate = policy_wins / n
    ref_winrate = ref_wins / n

    avg_pol_margin = sum(policy_margins) / n
    avg_ref_margin = sum(ref_margins) / n

    report = f"""### 📊 DPO Diagnostics

Preference pairs evaluated: **{n}**

**Policy model (after DPO)**  
- Win rate (chosen > rejected): **{policy_winrate:.2%}**  
- Avg logprob(chosen − rejected): **{avg_pol_margin:.3f}**

**Reference model (base)**  
- Win rate (chosen > rejected): **{ref_winrate:.2%}**  
- Avg logprob(chosen − rejected): **{avg_ref_margin:.3f}**

> A higher win rate and margin for the policy model compared to the reference model
> indicates that DPO training is successfully shifting the model toward your preferences.
"""
    return report


def generate_from_aligned_model(
    prompt: str,
    do_sample: bool,
    temperature: float,
    max_new_tokens: int,
) -> str:
    if not prompt.strip():
        return ""
    gen_config = build_generation_config(
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_k=20,
        top_p=0.9,
    )
    style_balanced = (
        "You are a helpful, careful assistant. "
        "Answer clearly and sensibly.\n\nUser: "
    )
    return generate_text(
        policy_model,
        prompt,
        gen_config,
        style_prefix=style_balanced,
    )


def on_model_change(
    model_name: str,
    _state_preferences: List[Dict],
):
    """
    When the user picks a new base model:
      - reload tokenizer + policy_model + ref_model
      - clear collected preferences (since they belong to previous model)
      - reset training status, 'last trained', and download list
    """
    status = load_base_model(model_name)
    empty_prefs: List[Dict] = []
    df = preferences_to_df(empty_prefs)
    reset_msg = (
        status
        + "\n\nPreferences cleared (new model = new preference data)."
    )
    last_trained_reset = "**Last trained:** (reset for new base model)"
    files_reset: List[str] = []
    # returns: model_status, prefs, pref_table_df, train_status, last_trained, files
    return reset_msg, empty_prefs, df, "", last_trained_reset, files_reset


# =========================================================
#  GRADIO UI
# =========================================================

with gr.Blocks() as demo:
    gr.Markdown(
        """
    # 🔧 DPO Playground – Preference Tuning on Different Models

    - Pick a **base model** from the dropdown.
    - Ask a question and generate two answers:
      - **A** = balanced / normal
      - **B** = creative / more extreme
    - Optionally write **your own ideal answer**.
    - Choose how to encode the preference (e.g. A over B, custom over A, etc.).
    - Collect several preferences and **train the model with DPO**.
    - Test how the aligned policy model behaves on new prompts.
    - Download the tuned model (LoRA adapter + tokenizer) after training.
    - Use **DPO diagnostics** to see if the aligned model prefers your chosen answers
      more often than the base model.
    """
    )

    state_preferences = gr.State([])

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=MODEL_CHOICES,
            value=DEFAULT_MODEL,
            label="Base model",
        )

    model_status = gr.Markdown(initial_status)

    # -----------------------------------------------------
    # Collect preferences tab
    # -----------------------------------------------------
    with gr.Tab("Collect preferences"):
        with gr.Row():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Ask anything...",
                lines=3,
            )

        gr.Markdown("### Generation settings for Answer A & B")

        with gr.Row():
            gen_do_sample = gr.Checkbox(
                value=True,
                label="Use sampling (do_sample)",
            )
            gen_temperature = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                value=0.8,
                step=0.05,
                label="Temperature",
            )
            gen_max_new_tokens = gr.Slider(
                minimum=4,
                maximum=256,
                value=128,
                step=4,
                label="Max new tokens",
            )

        generate_btn = gr.Button("Generate A & B")

        with gr.Row():
            answer_a_box = gr.Textbox(
                label="Answer A (balanced / normal)",
                lines=8,
            )
            answer_b_box = gr.Textbox(
                label="Answer B (creative / more extreme)",
                lines=8,
            )

        custom_answer_box = gr.Textbox(
            label="Your own ideal answer (optional)",
            lines=8,
            placeholder="If you want, write the answer you *wish* the model had given.",
        )

        preference_mode = gr.Radio(
            choices=[
                "Prefer A over B",
                "Prefer B over A",
                "Prefer custom over A",
                "Prefer custom over B",
            ],
            label="How should this preference be encoded?",
        )

        save_pref_btn = gr.Button("Save preference")

        pref_status = gr.Markdown("")
        pref_table = gr.Dataframe(
            headers=["prompt", "chosen", "rejected"],
            label="Collected preferences (for DPO training)",
            wrap=True,
        )

        generate_btn.click(
            fn=generate_candidates,
            inputs=[prompt_input, gen_do_sample, gen_temperature, gen_max_new_tokens],
            outputs=[answer_a_box, answer_b_box],
        )

        save_pref_btn.click(
            fn=save_preference,
            inputs=[
                prompt_input,
                answer_a_box,
                answer_b_box,
                custom_answer_box,
                preference_mode,
                state_preferences,
            ],
            outputs=[
                state_preferences,
                pref_table,
                pref_status,
            ],
        )

    # -----------------------------------------------------
    # Train & test tab
    # -----------------------------------------------------
    with gr.Tab("Train & test DPO model"):
        gr.Markdown(
            "Train the LoRA-adapted policy model using your preferences "
            "with **Direct Preference Optimization (DPO)**."
        )

        with gr.Row():
            num_epochs_slider = gr.Slider(
                minimum=1,
                maximum=5,
                step=1,
                value=1,
                label="Number of epochs",
            )
            lr_slider = gr.Slider(
                minimum=1e-5,
                maximum=5e-4,
                step=1e-5,
                value=1e-4,
                label="Learning rate",
            )
            beta_slider = gr.Slider(
                minimum=0.05,
                maximum=0.5,
                step=0.05,
                value=0.1,
                label="DPO beta (strength)",
            )

        train_btn = gr.Button("Train DPO model", variant="primary")
        train_status = gr.Markdown("")
        last_trained = gr.Markdown("**Last trained:** never")

        download_files = gr.Files(
            label="Trained model files (adapter + tokenizer)",
            interactive=False,
        )

        train_btn.click(
            fn=train_dpo_model,
            inputs=[
                state_preferences,
                num_epochs_slider,
                lr_slider,
                beta_slider,
            ],
            outputs=[train_status, last_trained, download_files],
        )

        gr.Markdown("## Try the current policy model")

        with gr.Row():
            test_do_sample = gr.Checkbox(
                value=False,
                label="Use sampling (do_sample) for test",
            )
            test_temperature = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                value=0.0,
                step=0.05,
                label="Temperature (test)",
            )
            test_max_new_tokens = gr.Slider(
                minimum=4,
                maximum=256,
                value=64,
                step=4,
                label="Max new tokens (test)",
            )

        test_prompt = gr.Textbox(
            label="Test prompt",
            placeholder="Ask something to see the aligned model...",
            lines=3,
        )
        test_btn = gr.Button("Generate from DPO policy model")
        test_answer = gr.Textbox(
            label="Policy model answer",
            lines=8,
        )

        test_btn.click(
            fn=generate_from_aligned_model,
            inputs=[
                test_prompt,
                test_do_sample,
                test_temperature,
                test_max_new_tokens,
            ],
            outputs=test_answer,
        )

        gr.Markdown("## 📈 DPO diagnostics")

        diag_btn = gr.Button("Compute preference win rates (policy vs base)")
        diag_output = gr.Markdown("")

        diag_btn.click(
            fn=dpo_diagnostics,
            inputs=[state_preferences],
            outputs=[diag_output],
        )

    # model change: reload + clear prefs + reset train status + last trained + downloads
    model_dropdown.change(
        fn=on_model_change,
        inputs=[model_dropdown, state_preferences],
        outputs=[
            model_status,
            state_preferences,
            pref_table,
            train_status,
            last_trained,
            download_files,
        ],
    )

if __name__ == "__main__":
    demo.queue().launch()
