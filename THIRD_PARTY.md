# Dependencies and research attribution

The MIT license covers this repository's original code and task generators.
Dependencies and downloaded model weights retain their own licenses; no model
weights are bundled in the repository or Python distributions.

- [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE)
- [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE)
- [Safetensors](https://github.com/huggingface/safetensors/blob/main/LICENSE)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE)
- [Harbor](https://github.com/harbor-framework/harbor/blob/main/LICENSE)
- [Qwen2.5-Coder-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)
  lists Apache-2.0. Other model revisions or families may have different terms.

The direct-edit track was inspired by
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
This implementation uses its own restricted tensor language. It does not reproduce
the article's harness or establish its reported results. No code from that
article is vendored here. The benchmark is not affiliated with the article's
authors, model providers, Harbor, Gymnasium or Hugging Face.
