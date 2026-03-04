"""
Enhanced LatexProducer: beam search with multi-head attention weight tracking.
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
        attn_history = []

        # step() returns (log_probs, state, attn)
        start_class_log_probabilities, state, start_attn = step(
            start_predictions, start_state
        )
        num_classes = start_class_log_probabilities.size()[1]

        start_top_log_probabilities, start_predicted_classes = (
            start_class_log_probabilities.topk(self.beam_size)
        )

        last_log_probabilities = start_top_log_probabilities
        predictions.append(start_predicted_classes)

        # At step 1, all beams share the same attention
        initial_attn = start_attn.unsqueeze(1).expand(
            batch_size, self.beam_size, *start_attn.shape[1:]
        )
        attn_history.append(initial_attn)

        # Expand state for beam
        for key, state_tensor in state.items():
            _, *last_dims = state_tensor.size()
            state[key] = (
                state_tensor.unsqueeze(1)
                .expand(batch_size, self.beam_size, *last_dims)
                .reshape(batch_size * self.beam_size, *last_dims)
            )

        log_probs_after_end = start_class_log_probabilities.new_full(
            (batch_size * self.beam_size, num_classes), float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.0

        for timestep in range(self.max_steps - 1):
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size)

            if (last_predictions == self._end_index).all():
                break

            class_log_probabilities, state, attn = step(last_predictions, state)

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

            # Filter attention by selected beams
            attn = attn.reshape(batch_size, self.beam_size, *attn.shape[1:])
            attn_bp = backpointer.view(
                batch_size, self.beam_size, *([1] * (len(attn.shape) - 2))
            ).expand_as(attn)
            restricted_attn = attn.gather(1, attn_bp)
            attn_history.append(restricted_attn)

            # Update state
            for key, state_tensor in state.items():
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
        reconstructed_attn = [attn_history[-1].unsqueeze(2)]

        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)
            reconstructed_predictions.append(cur_preds)

            cur_attn = (
                attn_history[timestep]
                .gather(
                    1,
                    cur_backpointers.view(batch_size, self.beam_size, 1, 1).expand_as(
                        attn_history[timestep]
                    ),
                )
                .unsqueeze(2)
            )
            reconstructed_attn.append(cur_attn)

            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)
        reconstructed_predictions.append(final_preds)

        final_attn = (
            attn_history[0]
            .gather(
                1,
                cur_backpointers.view(batch_size, self.beam_size, 1, 1).expand_as(
                    attn_history[0]
                ),
            )
            .unsqueeze(2)
        )
        reconstructed_attn.append(final_attn)

        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)
        all_attentions = torch.cat(list(reversed(reconstructed_attn)), 2)

        return all_predictions, last_log_probabilities, all_attentions


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
                if idx != END_TOKEN_ID:
                    result.append(self._id2sign[idx])
                else:
                    break
            results.append(" ".join(result))
        return results

    def _greedy_decoding(self, imgs):
        self.model.eval()
        imgs = imgs.to(self.device)

        enc_out = self.model.encoder(imgs)
        projected_key = self.model.decoder.mh_cross_attn.w_k(enc_out)
        projected_value = self.model.decoder.mh_cross_attn.w_v(enc_out)
        dec_states, o_t = self.model.decoder.init_decoder_states(enc_out)

        batch_size = imgs.size(0)
        formulas_idx = (
            torch.ones(batch_size, self.max_len, device=self.device).long()
            * PAD_TOKEN_ID
        )
        tgt = torch.ones(batch_size, 1, device=self.device).long() * START_TOKEN_ID
        all_attn = []
        with torch.no_grad():
            for t in range(self.max_len):
                dec_states, o_t, logit, attn_weights = self.model.decoder.forward_step(
                    dec_states, o_t, projected_key, projected_value, tgt
                )
                tgt = torch.argmax(logit, dim=1, keepdim=True)
                formulas_idx[:, t : t + 1] = tgt
                all_attn.append(attn_weights.detach().cpu())
        results = self._idx2formulas(formulas_idx)
        all_attn = torch.stack(all_attn, dim=1)
        return results, all_attn

    def _batch_beam_search(self, imgs):
        self.model.eval()
        imgs = imgs.to(self.device)

        enc_out = self.model.encoder(imgs)
        projected_key = self.model.decoder.mh_cross_attn.w_k(enc_out)
        projected_value = self.model.decoder.mh_cross_attn.w_v(enc_out)
        dec_states, o_t = self.model.decoder.init_decoder_states(enc_out)

        batch_size = imgs.size(0)
        start_predictions = (
            torch.ones(batch_size, 1, device=self.device).long() * START_TOKEN_ID
        )

        state = {
            "h_t": dec_states[0],
            "c_t": dec_states[1],
            "o_t": o_t,
            "projected_key": projected_key,
            "projected_value": projected_value,
        }

        with torch.no_grad():
            all_top_k_predictions, log_probabilities, all_attentions = (
                self._beam_search.search(start_predictions, state, self._take_step)
            )

        best_predictions = all_top_k_predictions[:, 0, :]
        best_attentions = all_attentions[:, 0, :, :, :]

        results = self._idx2formulas(best_predictions)
        return results, best_attentions

    def _take_step(self, last_predictions, state):
        h_t = state["h_t"]
        c_t = state["c_t"]
        o_t = state["o_t"]
        projected_key = state["projected_key"]
        projected_value = state["projected_value"]

        dec_states = (h_t, c_t)

        dec_states, o_t, logit, attn_weights = self.model.decoder.forward_step(
            dec_states, o_t, projected_key, projected_value, last_predictions
        )

        new_state = {
            "h_t": dec_states[0],
            "c_t": dec_states[1],
            "o_t": o_t,
            "projected_key": projected_key,
            "projected_value": projected_value,
        }

        return torch.log_softmax(logit, dim=1), new_state, attn_weights
