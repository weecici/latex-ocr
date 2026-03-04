import torch

START_TOKEN_ID = 0
PAD_TOKEN_ID = 1
END_TOKEN_ID = 2
UNK_TOKEN_ID = 3


class BeamSearch:
    """
    Implements the beam search algorithm for decoding the most likely sequences.

    Parameters
    ----------
    end_index : ``int``
        The index of the "stop" or "end" token in the target vocabulary.
    max_steps : ``int``, optional (default = 50)
        The maximum number of decoding steps to take, i.e. the maximum length
        of the predicted sequences.
    beam_size : ``int``, optional (default = 10)
        The width of the beam used.
    per_node_beam_size : ``int``, optional (default = beam_size)
        The maximum number of candidates to consider per node, at each step in the search.
        If not given, this just defaults to ``beam_size``. Setting this parameter
        to a number smaller than ``beam_size`` may give better results, 
        as it can introduce more diversity into the search. 
        See `Beam Search Strategies for Neural Machine Translation.
        Freitag and Al-Onaizan, 2017 <http://arxiv.org/abs/1702.01806>`_.
    """

    def __init__(self,
                 end_index: int,
                 max_steps: int = 50,
                 beam_size: int = 10,
                 per_node_beam_size: int = None) -> None:
        self._end_index = end_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size

    def search(self, start_predictions, start_state, step):
        """
        Given a starting state and a step function, apply beam search to find the
        most likely target sequences.

        Notes
        -----
        If your step function returns ``-inf`` for some log probabilities
        (like if you're using a masked log-softmax) then some of the "best"
        sequences returned may also have ``-inf`` log probability. Specifically
        this happens when the beam size is smaller than the number of actions
        with finite log probability (non-zero probability) returned by the step function.
        Therefore if you're using a mask you may want to check the results from ``search``
        and potentially discard sequences with non-finite log probability.

        Parameters
        ----------
        start_predictions : ``torch.Tensor``
            A tensor containing the initial predictions with shape ``(batch_size,)``.
            Usually the initial predictions are just the index of the "start" token
            in the target vocabulary.
        start_state : ``dict``
            The initial state passed to the ``step`` function. 
            Each value of the state dict should be a tensor of shape ``(batch_size, *)``, 
            where ``*`` means any other number of dimensions.
        step : ``function``
            A function that is responsible for computing the next most likely tokens,
            given the current state and the predictions from the last time step.
            The function should accept two arguments. The first being a tensor
            of shape ``(group_size,)``, representing the index of the predicted
            tokens from the last time step, and the second being the current state.
            The ``group_size`` will be ``batch_size * beam_size``, except in the initial
            step, for which it will just be ``batch_size``.
            The function is expected to return a tuple, where the first element
            is a tensor of shape ``(group_size, target_vocab_size)`` containing
            the log probabilities of the tokens for the next step, and the second
            element is the updated state. The tensor in the state should have shape
            ``(group_size, *)``, where ``*`` means any other number of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, log_probabilities)``, where ``predictions``
            has shape ``(batch_size, beam_size, max_steps)`` and ``log_probabilities``
            has shape ``(batch_size, beam_size)``.
        """
        batch_size = start_predictions.size()[0]

        # List of (batch_size, beam_size) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions = []

        # List of (batch_size, beam_size) tensors. One for each time step. None for
        # the first.  Stores the index n for the parent prediction, i.e.
        # predictions[t-1][i][n], that it came from.
        backpointers = []

        # Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from the
        # encoder) to the top `beam_size` decoder outputs. On the other hand,
        # within the main loop we are going from the `beam_size` elements of the
        # beam to `beam_size`^2 candidates from which we will select the top
        # `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        start_class_log_probabilities, state = step(
            start_predictions, start_state)

        num_classes = start_class_log_probabilities.size()[1]

        # shape: (batch_size, beam_size), (batch_size, beam_size)
        start_top_log_probabilities, start_predicted_classes = \
            start_class_log_probabilities.topk(self.beam_size)
        if self.beam_size == 1 and (start_predicted_classes == self._end_index).all():
            print("Empty sequences predicted. You may want to "
                  "increase the beam size or ensure "
                  "your step function is working properly.")
            return start_predicted_classes.unsqueeze(-1), start_top_log_probabilities

        # The log probabilities for the last time step.
        # shape: (batch_size, beam_size)
        last_log_probabilities = start_top_log_probabilities

        # shape: [(batch_size, beam_size)]
        predictions.append(start_predicted_classes)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        log_probs_after_end = start_class_log_probabilities.new_full(
            (batch_size * self.beam_size, num_classes),
            float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.

        # Set the same state for each element in the beam.
        for key, state_tensor in state.items():
            _, *last_dims = state_tensor.size()
            # shape: (batch_size * beam_size, *)
            state[key] = state_tensor.\
                unsqueeze(1).\
                expand(batch_size, self.beam_size, *last_dims).\
                reshape(batch_size * self.beam_size, *last_dims)

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[-1].reshape(
                batch_size * self.beam_size)

            # If every predicted token from the last step is `self._end_index`,
            # then we can stop early.
            if (last_predictions == self._end_index).all():
                break

            # Take a step. This get the predicted log probs of the next classes
            # and updates the state.
            # shape: (batch_size * beam_size, num_classes)
            class_log_probabilities, state = step(last_predictions, state)

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size,
                num_classes
            )

            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            # shape: (batch_size * beam_size, num_classes)
            cleaned_log_probabilities = torch.where(
                last_predictions_expanded == self._end_index,
                log_probs_after_end,
                class_log_probabilities
            )

            # shape (both): (batch_size * beam_size, per_node_beam_size)
            top_log_probabilities, predicted_classes = \
                cleaned_log_probabilities.topk(self.per_node_beam_size)

            # Here we expand the last log probabilities to (batch_size * beam_size, per_node_beam_size)
            # so that we can add them to the current log probs for this timestep.
            # This lets us maintain the log probability of each element on the beam.
            # shape: (batch_size * beam_size, per_node_beam_size)
            expanded_last_log_probabilities = last_log_probabilities.\
                unsqueeze(2).\
                expand(batch_size, self.beam_size, self.per_node_beam_size).\
                reshape(batch_size * self.beam_size, self.per_node_beam_size)

            # shape: (batch_size * beam_size, per_node_beam_size)
            summed_top_log_probabilities = top_log_probabilities + \
                expanded_last_log_probabilities

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_summed = summed_top_log_probabilities.\
                reshape(batch_size, self.beam_size * self.per_node_beam_size)

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.\
                reshape(batch_size, self.beam_size * self.per_node_beam_size)

            # Keep only the top `beam_size` beam indices.
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(
                self.beam_size)

            # Use the beam indices to extract the corresponding classes.
            # shape: (batch_size, beam_size)
            restricted_predicted_classes = reshaped_predicted_classes.gather(
                1, restricted_beam_indices)

            predictions.append(restricted_predicted_classes)

            # shape: (batch_size, beam_size)
            last_log_probabilities = restricted_beam_log_probs

            # The beam indices come from a `beam_size * per_node_beam_size` dimension where the
            # indices with a common ancestor are grouped together. Hence
            # dividing by per_node_beam_size gives the ancestor. (Note that this is integer
            # division as the tensor is a LongTensor.)
            # shape: (batch_size, beam_size)
            backpointer = restricted_beam_indices // self.per_node_beam_size

            backpointers.append(backpointer)

            # Keep only the pieces of the state tensors corresponding to the
            # ancestors created this iteration.
            for key, state_tensor in state.items():
                _, *last_dims = state_tensor.size()
                # shape: (batch_size, beam_size, *)
                expanded_backpointer = backpointer.\
                    view(batch_size, self.beam_size, *([1] * len(last_dims))).\
                    expand(batch_size, self.beam_size, *last_dims)

                # shape: (batch_size * beam_size, *)
                state[key] = state_tensor.\
                    reshape(batch_size, self.beam_size, *last_dims).\
                    gather(1, expanded_backpointer).\
                    reshape(batch_size * self.beam_size, *last_dims)

        if not torch.isfinite(last_log_probabilities).all():
            print("Infinite log probabilities encountered. "
                  "Some final sequences may not make sense. "
                  "This can happen when the beam size is "
                  "larger than the number of valid (non-zero "
                  "probability) transitions that the step function produces.")

        # Reconstruct the sequences.
        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = predictions[timestep].gather(
                1, cur_backpointers).unsqueeze(2)

            reconstructed_predictions.append(cur_preds)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep -
                                            1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(
            list(reversed(reconstructed_predictions)), 2)

        return all_predictions, last_log_probabilities
    

class LatexProducer(object):
    """
    Model wrapper, implementing batch greedy decoding and
    batch beam search decoding
    """

    def __init__(self, model, vocab, beam_size=5, max_len=64, use_cuda=True):
        """args:
            the path to model checkpoint
        """
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = model.to(self.device)
        self._sign2id = vocab.sign2id
        self._id2sign = vocab.id2sign
        self.max_len = max_len
        self.beam_size = beam_size
        self._beam_search = BeamSearch(END_TOKEN_ID, max_len, beam_size)

    def __call__(self, imgs):
        """args:
            imgs: images need to be decoded
            beam_size: if equal to 1, use greedy decoding
           returns:
            formulas list of batch_size length
        """
        if self.beam_size == 1:
            results = self._greedy_decoding(imgs)
        else:
            results = self._batch_beam_search(imgs)
        return results

    def _idx2formulas(self, formulas_idx):
        """convert formula id matrix to formulas list"""
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
        # first decoding step's input
        generated = torch.full((batch_size,), START_TOKEN_ID, dtype=torch.long, device=device)
        with torch.no_grad():
            for t in range(self.max_len):
                logit, _ = self.model.decoder(generated, enc_out, precomputed_enc_kv=precomputed_enc_kv)
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
            # 1. Encode một lần
            enc_out = self.model.encoder(imgs)
            precomputed_enc_kv = self.model.decoder.precompute_encoder_kv(enc_out)
    
            start_predictions = torch.full((batch_size,), START_TOKEN_ID, 
                                         dtype=torch.long, device=self.device)
    
            # 2. State chỉ chứa encoder KV + past self-attention KV
            start_state = {
                "enc_out": enc_out,
            }
            
            # Encoder KV (không đổi suốt quá trình)
            for i, (k, v) in enumerate(precomputed_enc_kv):
                start_state[f"enc_k_{i}"] = k
                start_state[f"enc_v_{i}"] = v
            
            # Decoder self-attention KV cache (khởi tạo rỗng)
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
        """
        QUAN TRỌNG: Mỗi step chỉ decode 1 token mới, tái sử dụng KV cache
        """
        enc_out = state["enc_out"]
        
        # Reconstruct encoder KV
        num_layers = sum(1 for key in state.keys() if key.startswith("enc_k_"))
        enc_kv = [(state[f"enc_k_{i}"], state[f"enc_v_{i}"]) for i in range(num_layers)]
        
        # Reconstruct decoder self-attention KV cache
        past_self_kvs = []
        for i in range(num_layers):
            k = state[f"self_k_{i}"]
            v = state[f"self_v_{i}"]
            if k is not None and v is not None:
                past_self_kvs.append((k, v))
            else:
                past_self_kvs.append(None)
        
        # CHỈ DECODE 1 TOKEN MỚI (không phải cả history!)
        input_tokens = last_predictions.unsqueeze(1)  # [group_size, 1]
        
        # Forward với KV cache
        logits, present_self_kvs = self._forward_with_cache(
            input_tokens, enc_out, enc_kv, past_self_kvs
        )
        
        log_probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)
        
        # Update state với KV mới
        new_state = {"enc_out": enc_out}
        
        # Encoder KV không đổi
        for i in range(num_layers):
            new_state[f"enc_k_{i}"] = state[f"enc_k_{i}"]
            new_state[f"enc_v_{i}"] = state[f"enc_v_{i}"]
        
        # Decoder self-attention KV được update
        for i in range(num_layers):
            if present_self_kvs[i] is not None:
                new_state[f"self_k_{i}"] = present_self_kvs[i][0]
                new_state[f"self_v_{i}"] = present_self_kvs[i][1]
            else:
                new_state[f"self_k_{i}"] = None
                new_state[f"self_v_{i}"] = None
        
        return log_probs, new_state
    
    
    def _forward_with_cache(self, tgt, enc_out, precomputed_enc_kv, past_self_kvs):
        """
        Custom forward pass với KV cache cho beam search
        """
        # Embedding
        tgt_emb = self.model.decoder.embedding(tgt) * self.model.decoder.embedding_scale
        tgt_emb = self.model.decoder.pos_encoding(tgt_emb)
        tgt_emb = self.model.decoder.dropout(tgt_emb)
        
        # Không cần mask vì chỉ decode 1 token
        x = tgt_emb
        present_self_kvs = []
        
        for i, layer in enumerate(self.model.decoder.layers):
            # Self-attention với cache
            attn_out, _, present_kv = layer.self_attn(
                x, x, x, 
                mask=None,  # Decode 1 token không cần mask
                past_kv=past_self_kvs[i],
                use_cache=True
            )
            x = layer.norm1(x + layer.dropout(attn_out))
            
            # Cross-attention với precomputed encoder KV
            attn_out, _, _ = layer.cross_attn(
                x, enc_out, enc_out,
                mask=None,
                precomputed_kv=precomputed_enc_kv[i]
            )
            x = layer.norm2(x + layer.dropout(attn_out))
            
            # FFN
            ffn_out = layer.ffn(x)
            x = layer.norm3(x + layer.dropout(ffn_out))
            
            present_self_kvs.append(present_kv)
        
        logits = self.model.decoder.fc_out(x)
        return logits, present_self_kvs