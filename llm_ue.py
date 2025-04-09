import numpy as np
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, GemmaForCausalLM, GenerationConfig
import torch
import signal
import threading

class TimeoutException(Exception):
    pass


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self):
        raise TimeoutException(self.error_message)

    def __enter__(self):
        self.timer = threading.Timer(self.seconds, self.handle_timeout)
        self.timer.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.cancel()

def temperature_scaling(logits, temperature=1):
    logits = np.array(logits)
    logits /= temperature

    # apply softmax
    try:
        logits -= logits.max()
    except:
        logits = logits
    logits = logits - np.log(np.sum(np.exp(logits)))
    smx = np.exp(logits)
    smx = [float(x) for x in smx]
    return smx

class LLModel():
    def __init__(self, model_name, model, tokenizer, estimator, model_adapter):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.generation_config = GenerationConfig.from_pretrained(self.model_name)
        self.model_adapter = model_adapter
        self.estimator = estimator
        self.n_alternatives = 10

    def _post_process_logits(self, out, model_inputs, eos_token_id):
        cut_logits = []
        cut_sequences = []
        cut_log_probs = []
        cut_alternatives = []
        lls = []

        all_logits = torch.stack(out.scores, dim=1)
        for i in range(len(model_inputs)):
            seq = out.sequences[i, model_inputs.shape[1] :].cpu()

            length = len(seq)
            for j in range(len(seq)):
                if seq[j] == eos_token_id:
                    length = j + 1
                    break

            tokens = seq[:length].tolist()
            cut_sequences.append(tokens)

            logits = all_logits[i, :length, :].cpu()
            cut_logits.append(logits.numpy())

            log_probs = logits.log_softmax(-1)
            cut_log_probs.append(log_probs.numpy())
            lls.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

            cut_alternatives.append([[] for _ in range(length)])
            for j in range(length):
                lt = logits[j, :].numpy()
                best_tokens = np.argpartition(lt, -self.n_alternatives)
                ln = len(best_tokens)
                best_tokens = best_tokens[ln - self.n_alternatives : ln]
                for t in best_tokens:
                    cut_alternatives[-1][j].append((t.item(), lt[t].item()))

                cut_alternatives[-1][j].sort(
                    key=lambda x: x[0] == cut_sequences[-1][j],
                    reverse=True,
                )

        result_dict = {
            "greedy_log_probs": cut_log_probs,
            "greedy_logits": cut_logits,
            "greedy_tokens": cut_sequences,
            "greedy_log_likelihoods": lls,
            "greedy_tokens_alternatives": cut_alternatives,
        }

        return result_dict

    def generate(self, prompt, return_logits=False):
        encoded = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        deps = {"model_inputs": encoded}
        model_inputs = deps["model_inputs"]

        model_inputs = (
            model_inputs
            if isinstance(model_inputs, torch.Tensor)
            else model_inputs["input_ids"]
        )

        args_generate = {"generation_config" : self.generation_config,
                        "max_new_tokens": 30}

        args_generate.update(
            {
                "return_dict_in_generate": True,
                "output_scores": True,
                "output_hidden_states": True,
            }
        )
        out = self.model_adapter.generate(model_inputs.to(self.device), **args_generate)

        result_dict = self._post_process_logits(
            out, model_inputs, self.tokenizer.eos_token_id
        )
        deps.update(result_dict)

        ue_score = self.estimator(deps)

        generated_text = [self.tokenizer.decode(g, skip_special_tokens=True) for g in deps['greedy_tokens']][0]

        if return_logits:
          with torch.no_grad():
            combined_ids = torch.cat([model_inputs.to(device), out.sequences[:, 1:]], dim=1)
            outputs = self.model_adapter(combined_ids)
            logits = outputs.logits[:, model_inputs.size(1):]

            # Create a mask to filter out special tokens
            special_tokens = self.tokenizer.all_special_ids
            mask = torch.ones_like(out.sequences[0, 1:], dtype=torch.bool)
            for token_id in special_tokens:
                mask &= (out.sequences[0, 1:] != token_id)

            filtered_logits = logits[:, mask]

            return (generated_text, filtered_logits, ue_score)
        else:
          return (generated_text, ue_score)

    def generate_batch(self, prompts, return_logits=False):
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        deps = {"model_inputs": encoded}
        model_inputs = deps["model_inputs"]

        model_inputs = (
            model_inputs
            if isinstance(model_inputs, torch.Tensor)
            else model_inputs["input_ids"]
        )

        args_generate = {"generation_config" : self.generation_config,
                        "max_new_tokens": 30}

        args_generate.update(
            {
                "return_dict_in_generate": True,
                "output_scores": True,
                "output_hidden_states": True,
            }
        )

        out = self.model_adapter.generate(model_inputs.to(self.device), **args_generate)

        result_dict = self._post_process_logits(
            out, model_inputs, self.tokenizer.eos_token_id
        )
        deps.update(result_dict)

        generated_text = [self.tokenizer.decode(g, skip_special_tokens=True) for g in deps['greedy_tokens']]

        ue_score = self.estimator(deps)

        if return_logits:
            logits = self._compute_logits(model_inputs.to(self.device), out.sequences)
            filtered_logits = self._filter_special_token_logits(out.sequences, logits)
            return (generated_text, filtered_logits, ue_score)
        else:
            return (generated_text, ue_score)

    def _compute_logits(self, input_ids, generated_ids):
      with torch.no_grad():
        combined_ids = torch.cat([input_ids, generated_ids[:, 1:]], dim=1)
        outputs = self.model_adapter(combined_ids)
        logits = outputs.logits[:, input_ids.size(1):]
      return logits

    def _filter_special_token_logits(self, generated_ids, logits):
      special_tokens = self.tokenizer.all_special_ids
      batch_size, seq_length = generated_ids.shape
      mask = torch.ones((batch_size, seq_length - 1), dtype=torch.bool, device=logits.device)
      for token_id in special_tokens:
          mask &= (generated_ids[:, 1:] != token_id)
      filtered_logits = torch.stack([logit[mask[i]] for i, logit in enumerate(logits)])
      return filtered_logits

    def filter_logits(self, logits, words, use_softmax=True):
      # Initialize an empty list to collect all token IDs
      token_ids = []
      # Tokenize each word and convert to IDs
      for word in words:
          word_tokens = self.tokenizer.tokenize(word)
          word_token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
          token_ids.extend(word_token_ids)

      token_ids = [t.item() for t in torch.tensor(token_ids, device=self.device)]

      count_tokens = dict(Counter(token_ids).most_common())
      token_ids_target = [key for key in count_tokens.keys() if count_tokens[key] == 1]
      filtered_logits = [logits[t].item() for t in token_ids_target]
      if use_softmax:
          filtered_logits = temperature_scaling(filtered_logits)
      return dict(zip(words, filtered_logits))
