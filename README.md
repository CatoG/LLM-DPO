---
title: DPO Demo
emoji: 📚
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 6.0.2
app_file: app.py
pinned: false
short_description: Testing DPO for finetuning models
---
A test / demo application playground for DPO Preference Tuning on different LLM models.
Running on Huggingspace:
https://huggingface.co/spaces/CatoG/DPO_Demo

Allows for LLM model selection, preference tuning of LLM responses, model response tuning with LoRA and Direct Preference Optimization (DPO).
Tuned model / policies can be downloaded for further use.

This project is an interactive Direct Preference Optimization (DPO) playground for experimenting with real LLM behavior-tuning. The app lets you load a variety of open models, generate multiple candidate answers, and explicitly encode human preferences (chosen vs. rejected responses) through an intuitive Gradio interface.

Using these preference pairs, the app trains a LoRA-adapted policy model against a frozen reference model, shifting the model’s behavior toward your desired style, tone, or reasoning pattern. You can explore how DPO changes alignment by collecting preferences, running training rounds, and immediately testing the tuned policy model on new prompts.

Purpose: Experimental tool for understanding alignment, safety, and model personalization techniques, without requiring deep ML infrastructure. It supports multiple models, adjustable generation parameters, preference visualization, and downloadable tuned LoRA adapters for further use.
