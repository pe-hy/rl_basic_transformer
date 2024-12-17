# custom_config.py

from litgpt.config import configs, Config, name_to_config

# Define your custom model configuration
custom_config = dict(
    name="SmolLM2-135M{}",
    hf_config=dict(org="HuggingFaceTB", name="SmolLM2-135M{}"),
    block_size=4096,
    vocab_size=410,
    padded_vocab_size=410,
    n_layer=30,
    n_head=9,
    n_embd=576,
    n_query_groups=3,
    rotary_percentage=1.0,
    parallel_residual=False,
    bias=False,
    norm_class_name="RMSNorm",
    mlp_class_name="LLaMAMLP",
    intermediate_size=1024,
    rope_base=100000,
    norm_eps=1e-5,
)

# Add configurations for both base and instruct versions
for kind in ("", "-Instruct"):
    config_copy = custom_config.copy()
    config_copy["name"] = custom_config["name"].format(kind)
    config_copy["hf_config"]["name"] = custom_config["hf_config"]["name"].format(kind)

    # Add to configs list
    configs.append(config_copy)
    # Update name_to_config dictionary
    name_to_config[config_copy["name"]] = config_copy
