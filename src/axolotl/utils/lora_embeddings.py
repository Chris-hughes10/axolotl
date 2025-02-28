"""
helpers for lora embeddings
"""


def get_linear_embedding_layers(model_type):
    """
    returns the linear embedding layers needed for loras, dependent on the model arch
    """
    if model_type == "phi-msft":
        return ["embd.wte", "lm_head.linear"]
    if model_type == "gpt_neox":
        return ["embed_in", "embed_out"]
    return ["embed_tokens", "lm_head"]
