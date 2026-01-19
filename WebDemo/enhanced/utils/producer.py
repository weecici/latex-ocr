import torch

START_TOKEN_ID = 0
PAD_TOKEN_ID = 1
END_TOKEN_ID = 2
UNK_TOKEN_ID = 3


class BeamSearch:
    """
    Implements the beam search algorithm with attention weight tracking.
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
        Dự đoán chuỗi và thu thập attention weights.
        """
        batch_size = start_predictions.size()[0]
        predictions = []
        backpointers = []
        # Danh sách lưu attention weights tại mỗi bước
        attn_history = [] 

        # Bước 1: Tính toán timestep đầu tiên [cite: 21]
        # step() giờ đây trả về (log_probs, state, attn)
        start_class_log_probabilities, state, start_attn = step(start_predictions, start_state)
        num_classes = start_class_log_probabilities.size()[1]

        # Lấy top k kết quả đầu tiên [cite: 22]
        start_top_log_probabilities, start_predicted_classes = start_class_log_probabilities.topk(self.beam_size)
        
        last_log_probabilities = start_top_log_probabilities
        predictions.append(start_predicted_classes)

        # Lưu attention đầu tiên (cần lọc theo top_k index)
        # start_attn shape: (batch_size, num_heads, seq_len) -> chuyển thành (batch_size, beam_size, ...)
        # Ở bước 1, tất cả các beam trong một mẫu dùng chung attention của mẫu đó
        initial_attn = start_attn.unsqueeze(1).expand(batch_size, self.beam_size, *start_attn.shape[1:])
        attn_history.append(initial_attn)

        # Chuẩn bị state cho các bước sau (expand cho beam size) [cite: 25]
        for key, state_tensor in state.items():
            _, *last_dims = state_tensor.size()
            state[key] = state_tensor.unsqueeze(1).expand(batch_size, self.beam_size, *last_dims).reshape(batch_size * self.beam_size, *last_dims)

        # Log prob để ép kết thúc sau token END [cite: 24, 25]
        log_probs_after_end = start_class_log_probabilities.new_full((batch_size * self.beam_size, num_classes), float("-inf"))
        log_probs_after_end[:, self._end_index] = 0.

        # Bước 2: Vòng lặp giải mã [cite: 26]
        for timestep in range(self.max_steps - 1):
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size)

            if (last_predictions == self._end_index).all(): 
                break

            # Gọi step để lấy log_probs và attention của bước này [cite: 30]
            class_log_probabilities, state, attn = step(last_predictions, state)

            # Xử lý các chuỗi đã kết thúc [cite: 32]
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(batch_size * self.beam_size, num_classes)
            cleaned_log_probabilities = torch.where(last_predictions_expanded == self._end_index, log_probs_after_end, class_log_probabilities)

            # Tính điểm số cộng dồn [cite: 36]
            top_log_probabilities, predicted_classes = cleaned_log_probabilities.topk(self.per_node_beam_size)
            expanded_last_log_probabilities = last_log_probabilities.unsqueeze(2).expand(batch_size, self.beam_size, self.per_node_beam_size).reshape(batch_size * self.beam_size, self.per_node_beam_size)
            summed_top_log_probabilities = top_log_probabilities + expanded_last_log_probabilities

            # Lọc ra top beam_size tốt nhất từ beam_size * per_node_beam_size ứng viên [cite: 38]
            reshaped_summed = summed_top_log_probabilities.reshape(batch_size, self.beam_size * self.per_node_beam_size)
            restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(self.beam_size)

            # Lấy predicted classes tương ứng [cite: 39]
            reshaped_predicted_classes = predicted_classes.reshape(batch_size, self.beam_size * self.per_node_beam_size)
            restricted_predicted_classes = reshaped_predicted_classes.gather(1, restricted_beam_indices)
            predictions.append(restricted_predicted_classes)

            last_log_probabilities = restricted_beam_log_probs

            # Tính backpointer để biết nhánh hiện tại đến từ node nào bước trước [cite: 42]
            backpointer = restricted_beam_indices // self.per_node_beam_size
            backpointers.append(backpointer)

            # Lưu và lọc attention của bước này theo các nhánh được chọn
            # attn shape: (batch_size * beam_size, num_heads, seq_len)
            attn = attn.reshape(batch_size, self.beam_size, *attn.shape[1:])
            # Mở rộng backpointer để gather attention
            attn_bp = backpointer.view(batch_size, self.beam_size, *([1] * (len(attn.shape)-2))).expand_as(attn)
            restricted_attn = attn.gather(1, attn_bp)
            attn_history.append(restricted_attn)

            # Cập nhật state [cite: 44, 45]
            for key, state_tensor in state.items():
                _, *last_dims = state_tensor.size()
                expanded_backpointer = backpointer.view(batch_size, self.beam_size, *([1] * len(last_dims))).expand(batch_size, self.beam_size, *last_dims)
                state[key] = state_tensor.reshape(batch_size, self.beam_size, *last_dims).gather(1, expanded_backpointer).reshape(batch_size * self.beam_size, *last_dims)

        # Bước 3: Tái cấu trúc chuỗi (Reconstruction) [cite: 47]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]
        reconstructed_attn = [attn_history[-1].unsqueeze(2)] # Thu thập attention của bước cuối

        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # Lấy predictions [cite: 48]
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)
            reconstructed_predictions.append(cur_preds)

            # Lấy attention tương ứng qua backpointer
            cur_attn = attn_history[timestep].gather(1, cur_backpointers.view(batch_size, self.beam_size, 1, 1).expand_as(attn_history[timestep])).unsqueeze(2)
            reconstructed_attn.append(cur_attn)

            # Cập nhật backpointer lùi về trước [cite: 48]
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        # Bước đầu tiên 
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)
        reconstructed_predictions.append(final_preds)
        
        final_attn = attn_history[0].gather(1, cur_backpointers.view(batch_size, self.beam_size, 1, 1).expand_as(attn_history[0])).unsqueeze(2)
        reconstructed_attn.append(final_attn)

        # Tổng hợp kết quả
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)
        all_attentions = torch.cat(list(reversed(reconstructed_attn)), 2) # (B, K, Max_Len, Heads, Seq_Len)

        return all_predictions, last_log_probabilities, all_attentions
    

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
        # storing decoding results
        formulas_idx = torch.ones(
            batch_size, self.max_len, device=self.device).long() * PAD_TOKEN_ID
        # first decoding step's input
        tgt = torch.ones(
            batch_size, 1, device=self.device).long() * START_TOKEN_ID
        with torch.no_grad():
            for t in range(self.max_len):
                dec_states, o_t, logit = self.model.decoder.forward_step(
                    dec_states, o_t, projected_key, projected_value, tgt
                )

                tgt = torch.argmax(logit, dim=1, keepdim=True)
                formulas_idx[:, t:t + 1] = tgt
        results = self._idx2formulas(formulas_idx)
        return results

    def _batch_beam_search(self, imgs):
            self.model.eval()
            imgs = imgs.to(self.device)
         
            # Encoding
            enc_out = self.model.encoder(imgs)  # [batch_size, H*W, OUT_C]
            projected_key = self.model.decoder.mh_cross_attn.w_k(enc_out)
            projected_value = self.model.decoder.mh_cross_attn.w_v(enc_out)
            # Init Decoder
            dec_states, o_t = self.model.decoder.init_decoder_states(enc_out)
            
            batch_size = imgs.size(0)
            start_predictions = torch.ones(
                batch_size, 1, device=self.device).long() * START_TOKEN_ID
            
            # Đóng gói state vào dict
            state = {}
            state['h_t'] = dec_states[0] # LSTM hidden
            state['c_t'] = dec_states[1] # LSTM cell
            state['o_t'] = o_t
            state['projected_key'] = projected_key
            state['projected_value'] = projected_value
            
            # Gọi Beam Search
            # Trả về: [B, K, Max_Len], [B, K]
            with torch.no_grad():
                all_top_k_predictions, log_probabilities, all_attentions = self._beam_search.search(
                    start_predictions, state, self._take_step)
    
            # Lấy kết quả tốt nhất (Beam index 0 vì đã sort trong beam search)
            best_predictions = all_top_k_predictions[:, 0, :]
            best_attentions = all_attentions[:, 0, :, :, :]
            # Chuyển thành text
            results = self._idx2formulas(best_predictions)
            return results, best_attentions

    def _take_step(self, last_predictions, state):
            """
            Hàm này chạy forward step cho [Batch * Beam_Size] mẫu cùng lúc
            """
            # Unpack state
            # Lưu ý: Các tensor này đã được BeamSearch expand thành [B*K, ...]
            h_t = state['h_t']
            c_t = state['c_t']
            o_t = state['o_t']
            projected_key = state['projected_key']  
            projected_value = state['projected_value']
            
            dec_states = (h_t, c_t)
            
            # Model Forward Step
            # last_predictions shape: [B*K, 1]
            dec_states, o_t, logit, attn_weights = self.model.decoder.forward_step(
                dec_states, o_t, projected_key, projected_value, last_predictions
            )
    
            # Cập nhật lại state mới vào dict để trả về cho BeamSearch quản lý
            new_state = {}
            new_state['h_t'] = dec_states[0]
            new_state['c_t'] = dec_states[1]
            new_state['o_t'] = o_t
            new_state['projected_key'] = projected_key
            new_state['projected_value'] = projected_value 
            
            # Trả về log_softmax để cộng dồn điểm số
            return torch.log_softmax(logit, dim=1), new_state, attn_weights