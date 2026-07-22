# Prompt and LoRA span format

3DreamBooth and Joint configs mark the subject phrase directly inside the prompt with square brackets. There is no separate `text_lora_spans` setting in public configs.

```yaml
train_prompts:
  - A video of a [rhs plushie].

validation_prompts:
  - A video of a [rhs plushie] on a beach at golden hour.
```

The canonical training template is:

```text
A video of a [v] [class].
```

`[v] [class]` describes the two conceptual parts of the marked phrase: a rare identifier and its class word. In an actual config, place both inside one pair of square brackets:

```text
A video of a [rhs plushie].
```

The config runner translates this markup differently by stage:

- Training prompt: `A video of a [rhs plushie].` becomes `A video of a rhs plushie.` and LoRA is applied to the full training prompt.
- Training-time validation: the brackets are removed from the text, and `rhs plushie` is passed automatically as `validation_lora_spans`.
- Standalone validation: the brackets are removed from the text, and `rhs plushie` is passed automatically as `text_lora_spans`.

This keeps the text sent to the tokenizer clean while making the validation-only LoRA region visible in the prompt itself.

Multiple marked spans are supported:

```yaml
prompt: A [rhs plushie] sitting on a [kad floor].
```

This produces the clean prompt `A rhs plushie sitting on a kad floor.` and the spans `rhs plushie`, `kad floor`. Unbalanced brackets are rejected before model loading. A 3DreamBooth or Joint validation prompt must contain at least one marked span.

3Dapter prompts do not use text LoRA, so they do not require bracket markup.
