import transformers
from score.model.modify_llama_score import llama_attn_forward_score_modify, llama_model_forward_score_modify
from score.model.modify_mistral_score import mistral_model_forward_score, mistral_attn_forward_score
from score.model.modify_qwen2_score import qwen2_model_forward_score, qwen2_attn_forward_score

def replace_flashllama_attn_with_scoreattn():
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward_score_modify
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_attn_forward_score_modify


def replace_flashmistral_attn_with_scoreattn():
    transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward_score
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_attn_forward_score


def replace_flashqwen2_attn_with_scoreattn():
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen2_model_forward_score
    transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_attn_forward_score



