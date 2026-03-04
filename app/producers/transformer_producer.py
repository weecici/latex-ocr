"""
Transformer LatexProducer: beam search with KV-cache, no attention returned.
"""

import torch

START_TOKEN_ID = 0
PAD_TOKEN_ID = 1
END_TOKEN_ID = 2
UNK_TOKEN_ID = 3


class BeamSearch:
    def __init__(self, end_index, max_steps=50, beam_size=10, per_node_beam_size=None):
        self._end_index = end_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size

    def search(self, start_predictions, start_state, step):
        batch_size = start_predictions.size()[0]
        predictions = []
        backpointers = []

        start_class_log_probabilities, state = step(start_predictions, start_state)
        num_classes = start_class_log_probabilities.size()[1]

        start_top_log_probabilities, start_predicted_classes = (
            start_class_log_probabilities.topk(self.beam_size)
        )
        if self.beam_size == 1 and (start_predicted_classes == self._end_index).all():
            return start_predicted_classes.unsqueeze(-1), start_top_log_probabilities

        last_log_probabilities = start_top_log_probabilities
        predictions.append(start_predicted_classes)

        log_probs_after_end = start_class_log_probabilities.new_full(
            (batch_size * self.beam_size, num_classes), float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.0

        for key, state_tensor in state.items():
            if state_tensor is None:
                continue
            _, *last_dims = state_tensor.size()
            state[key] = (
                state_tensor.unsqueeze(1)
                .expand(batch_size, self.beam_size, *last_dims)
                .reshape(batch_size * self.beam_size, *last_dims)
            )

        for timestep in range(self.max_steps - 1):
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size)

            if (last_predictions == self._end_index).all():
                break

            class_log_probabilities, state = step(last_predictions, state)

            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size, num_classes
            )
            cleaned_log_probabilities = torch.where(
                last_predictions_expanded == self._end_index,
                log_probs_after_end,
                class_log_probabilities,
            )

            top_log_probabilities, predicted_classes = cleaned_log_probabilities.topk(
                self.per_node_beam_size
            )
            expanded_last_log_probabilities = (
                last_log_probabilities.unsqueeze(2)
                .expand(batch_size, self.beam_size, self.per_node_beam_size)
                .reshape(batch_size * self.beam_size, self.per_node_beam_size)
            )
            summed_top_log_probabilities = (
                top_log_probabilities + expanded_last_log_probabilities
            )

            reshaped_summed = summed_top_log_probabilities.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )
            reshaped_predicted_classes = predicted_classes.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )

            restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(
                self.beam_size
            )
            restricted_predicted_classes = reshaped_predicted_classes.gather(
                1, restricted_beam_indices
            )
            predictions.append(restricted_predicted_classes)
            last_log_probabilities = restricted_beam_log_probs

            backpointer = restricted_beam_indices // self.per_node_beam_size
            backpointers.append(backpointer)

            for key, state_tensor in state.items():
                if state_tensor is None:
                    continue
                _, *last_dims = state_tensor.size()
                expanded_backpointer = backpointer.view(
                    batch_size, self.beam_size, *([1] * len(last_dims))
                ).expand(batch_size, self.beam_size, *last_dims)
                state[key] = (
                    state_tensor.reshape(batch_size, self.beam_size, *last_dims)
                    .gather(1, expanded_backpointer)
                    .reshape(batch_size * self.beam_size, *last_dims)
                )

        # Reconstruct
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)
            reconstructed_predictions.append(cur_preds)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)
        reconstructed_predictions.append(final_preds)

        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)
        return all_predictions, last_log_probabilities


class LatexProducer:
    def __init__(self, model, vocab, beam_size=5, max_len=64, use_cuda=True):
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = model.to(self.device)
        self._sign2id = vocab.sign2id
        self._id2sign = vocab.id2sign
        self.max_len = max_len
        self.beam_size = beam_size
        self._beam_search = BeamSearch(END_TOKEN_ID, max_len, beam_size)

    def __call__(self, imgs):
        if self.beam_size == 1:
            return self._greedy_decoding(imgs)
        return self._batch_beam_search(imgs)

    def _idx2formulas(self, formulas_idx):
        results = []
        for formula in formulas_idx:
            formula = formula.tolist()
            result = []
            for idx in formula:
                if idx == END_TOKEN_ID:
                    break
                if idx != START_TOKEN_ID and idx != PAD_TOKEN_ID:
                    result.append(self._id2sign[idx])
            results.append(" ".join(result))
        return results

    def _greedy_decoding(self, imgs):
        self.model.eval()
        imgs = imgs.to(self.device)

        enc_out = self.model.encoder(imgs)
        precomputed_enc_kv = self.model.decoder.precompute_encoder_kv(enc_out)

        batch_size = imgs.size(0)
        generated = torch.full(
            (batch_size, 1), START_TOKEN_ID, dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            for t in range(self.max_len):
                logit, _ = self.model.decoder(
                    generated, enc_out, precomputed_enc_kv=precomputed_enc_kv
                )
                next_token = torch.argmax(logit[:, -1:, :], dim=-1)
                generated = torch.cat([generated, next_token], dim=1)
                if (next_token == END_TOKEN_ID).all():
                    break
        return self._idx2formulas(generated)

    def _batch_beam_search(self, imgs):
        self.model.eval()
        imgs = imgs.to(self.device)
        batch_size = imgs.size(0)

        with torch.no_grad():
            enc_out = self.model.encoder(imgs)
            precomputed_enc_kv = self.model.decoder.precompute_encoder_kv(enc_out)

            start_predictions = torch.full(
                (batch_size,), START_TOKEN_ID, dtype=torch.long, device=self.device
            )

            start_state = {"enc_out": enc_out}

            for i, (k, v) in enumerate(precomputed_enc_kv):
                start_state[f"enc_k_{i}"] = k
                start_state[f"enc_v_{i}"] = v

            num_layers = len(self.model.decoder.layers)
            for i in range(num_layers):
                start_state[f"self_k_{i}"] = None
                start_state[f"self_v_{i}"] = None

            predictions, log_probs = self._beam_search.search(
                start_predictions, start_state, self._take_step
            )

            best_predictions = predictions[:, 0, :]

        return self._idx2formulas(best_predictions)

    def _take_step(self, last_predictions, state):
        enc_out = state["enc_out"]

        num_layers = sum(1 for key in state.keys() if key.startswith("enc_k_"))
        enc_kv = [(state[f"enc_k_{i}"], state[f"enc_v_{i}"]) for i in range(num_layers)]

        past_self_kvs = []
        for i in range(num_layers):
            k = state[f"self_k_{i}"]
            v = state[f"self_v_{i}"]
            if k is not None and v is not None:
                past_self_kvs.append((k, v))
            else:
                past_self_kvs.append(None)

        input_tokens = last_predictions.unsqueeze(1)

        logits, present_self_kvs = self._forward_with_cache(
            input_tokens, enc_out, enc_kv, past_self_kvs
        )

        log_probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)

        new_state = {"enc_out": enc_out}

        for i in range(num_layers):
            new_state[f"enc_k_{i}"] = state[f"enc_k_{i}"]
            new_state[f"enc_v_{i}"] = state[f"enc_v_{i}"]

        for i in range(num_layers):
            if present_self_kvs[i] is not None:
                new_state[f"self_k_{i}"] = present_self_kvs[i][0]
                new_state[f"self_v_{i}"] = present_self_kvs[i][1]
            else:
                new_state[f"self_k_{i}"] = None
                new_state[f"self_v_{i}"] = None

        return log_probs, new_state

    def _forward_with_cache(self, tgt, enc_out, precomputed_enc_kv, past_self_kvs):
        tgt_emb = self.model.decoder.embedding(tgt) * self.model.decoder.embedding_scale
        tgt_emb = self.model.decoder.pos_encoding(tgt_emb)
        tgt_emb = self.model.decoder.dropout(tgt_emb)

        x = tgt_emb
        present_self_kvs = []

        for i, layer in enumerate(self.model.decoder.layers):
            attn_out, _, present_kv = layer.self_attn(
                x, x, x, mask=None, past_kv=past_self_kvs[i], use_cache=True
            )
            x = layer.norm1(x + layer.dropout(attn_out))

            attn_out, _, _ = layer.cross_attn(
                x, enc_out, enc_out, mask=None, precomputed_kv=precomputed_enc_kv[i]
            )
            x = layer.norm2(x + layer.dropout(attn_out))

            ffn_out = layer.ffn(x)
            x = layer.norm3(x + layer.dropout(ffn_out))

            present_self_kvs.append(present_kv)

        logits = self.model.decoder.fc_out(x)
        return logits, present_self_kvs
