#!/usr/bin/env python3
import argparse
from pathlib import Path


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-py", required=True)
    a = p.parse_args()

    path = Path(a.model_py)
    text = path.read_text(encoding="utf-8")

    old = """    def parse_input(self, text, prompt_text, prompt_speech_tokens):
        total_text = f\"{prompt_text}{text}\"
        prompt = self.prompt_template.format(input_text=total_text)
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.int32)
        input_ids = torch.cat([input_ids, prompt_speech_tokens], dim=1)
        return input_ids
"""
    new = """    def parse_input(self, text, prompt_text, prompt_speech_tokens):
        # Navoiy instruct2: do NOT append prompt speech tokens to the LLM.
        # Reference voice conditioning is handled by cached Token2Wav features.
        total_text = f\"{prompt_text}{text}\"
        prompt = self.prompt_template.format(input_text=total_text)
        input_ids = self.tokenizer.encode(prompt)
        return torch.tensor([input_ids], dtype=torch.int32)
"""
    text = replace_once(text, old, new, "parse_input patch")

    old = """                reference_text = self.default_spk_info[\"prompt_text\"]
                prompt_speech_tokens = self.default_spk_info[\"speech_token\"] + ORIGINAL_VOCAB_SIZE
                prompt_speech_feat = None
                prompt_spk_embedding = None
"""
    new = """                # Fixed cached voice. For Navoiy instruct2 the LLM prompt is the
                # selected emotion instruction, not reference transcript/audio tokens.
                reference_text = os.environ[\"NAVOIY_INSTRUCTION\"]
                prompt_speech_tokens = self.default_spk_info[\"speech_token\"] + ORIGINAL_VOCAB_SIZE
                prompt_speech_feat = None
                prompt_spk_embedding = None
"""
    text = replace_once(text, old, new, "cached instruct2 patch")


    # V6C Step 2.2: route ONLY TRT-LLM speech-token generation over localhost gRPC.
    # The Triton server continues to serve both models; Token2Wav still uses
    # the original BLS request. Preserve input_dict, tokenizer and sampling.
    # Patch the verified upstream forward_llm block at build time.
    marker_start = "        # Convert inputs to Triton tensors\n"
    marker_end = "    def forward_audio_tokenizer(self, wav, wav_len):\n"
    start = text.find(marker_start)
    end = text.find(marker_end, start)
    if start < 0 or end < 0 or text.count(marker_start) != 1 or text.count(marker_end) != 1:
        raise RuntimeError("Could not identify pinned upstream forward_llm BLS block")
    original_llm = text[start:end]
    if 'llm_request.exec(decoupled=self.decoupled)' not in original_llm:
        raise RuntimeError("Expected BLS invocation absent; refusing to patch")
    grpc_llm = """        # Loopback gRPC replaces the native BLS iterator that can stall after Token2Wav.
        if not self.decoupled:
            raise pb_utils.TritonModelException('Navoiy gRPC path requires decoupled mode')
        import queue as _queue
        import tritonclient.grpc as _grpc
        from tritonclient.utils import np_to_triton_dtype as _trt_dtype
        _response_queue = _queue.Queue()
        _request_id = str(uuid4())
        _deadline = time.monotonic() + float(os.environ.get('NAVOIY_LLM_TIMEOUT_S', '35'))
        _success = False
        _generated_tokens = 0
        _responses = 0

        def _on_response(result, error):
            # Triton calls with result= and error= keyword arguments.
            try:
                if error is not None:
                    _response_queue.put(('error', str(error)))
                    return
                response = result.get_response()
                final_param = response.parameters.get('triton_final_response')
                final = bool(final_param.bool_param) if final_param is not None else False
                _response_queue.put(('result', result, final))
            except BaseException as exc:
                _response_queue.put(('error', repr(exc)))

        _client = _grpc.InferenceServerClient(
            url=os.environ.get('NAVOIY_LLM_GRPC_ENDPOINT', '127.0.0.1:8001'),
            verbose=False,
        )
        try:
            _inputs = []
            for _name, _value in input_dict.items():
                _value = np.ascontiguousarray(_value)
                _inp = _grpc.InferInput(_name, list(_value.shape), _trt_dtype(_value.dtype))
                _inp.set_data_from_numpy(_value)
                _inputs.append(_inp)
            _client.start_stream(callback=_on_response)
            _client.async_stream_infer(
                model_name='tensorrt_llm',
                inputs=_inputs,
                outputs=[_grpc.InferRequestedOutput('output_ids'),
                         _grpc.InferRequestedOutput('sequence_length')],
                request_id=_request_id,
                enable_empty_final_response=True,
            )
            self.logger.log_info(f'NAVOIY_GRPC_LLM_BEGIN request={_request_id} input_tokens={input_ids.shape[1]}')
            while True:
                _remaining = _deadline - time.monotonic()
                if _remaining <= 0:
                    raise TimeoutError(f'localhost gRPC TRT timed out request={_request_id} tokens={_generated_tokens}')
                try:
                    _item = _response_queue.get(timeout=_remaining)
                except _queue.Empty:
                    raise TimeoutError(f'localhost gRPC TRT did not complete request={_request_id} tokens={_generated_tokens}')
                if _item[0] == 'error':
                    raise RuntimeError(f'localhost gRPC TRT failure request={_request_id}: {_item[1]}')
                _, _result, _final = _item
                _actual_id = _result.get_response().id
                if _actual_id != _request_id:
                    raise RuntimeError(f'wrong TRT request_id: {_actual_id} != {_request_id}')
                if _final:
                    if _generated_tokens == 0:
                        raise RuntimeError('TRT returned final response with zero speech tokens')
                    _success = True
                    self.logger.log_info(f'NAVOIY_GRPC_LLM_DONE request={_request_id} tokens={_generated_tokens} responses={_responses}')
                    return
                _out = _result.as_numpy('output_ids')
                _lens = _result.as_numpy('sequence_length')
                if _out is None or _lens is None:
                    raise RuntimeError('TRT response missing output_ids or sequence_length')
                _len = int(_lens[0][0])
                _tokens = np.asarray(_out[0][0][:_len], dtype=np.int32).reshape(-1).copy()
                _responses += 1
                _generated_tokens += _tokens.size
                if _responses == 1:
                    self.logger.log_info(f'NAVOIY_GRPC_LLM_FIRST request={_request_id} n={_tokens.size}')
                if _tokens.size:
                    yield _tokens
        finally:
            try:
                _client.stop_stream(cancel_requests=not _success)
            except Exception as exc:
                self.logger.log_warn(f'NAVOIY_GRPC_LLM_CLOSE_ERROR {type(exc).__name__}: {exc}')
            try:
                _client.close()
            except Exception:
                pass
"""
    text = text[:start] + grpc_llm + text[end:]

    # Ensure a failed gRPC worker produces an error rather than leaving the
    # main CosyVoice execution loop spinning forever. This is error handling,
    # not a change to the successful token or audio paths.
    old = """    def _llm_gen_thread(self, generated_ids_iter, semantic_token_ids_arr, llm_is_done_flag):
        for generated_ids in generated_ids_iter:
            generated_ids = generated_ids.tolist()
            if len(generated_ids) == 0:
                break
            semantic_token_ids_arr.extend(generated_ids)
        llm_is_done_flag[0] = True
"""
    new = """    def _llm_gen_thread(self, generated_ids_iter, semantic_token_ids_arr, llm_is_done_flag, llm_error):
        try:
            for generated_ids in generated_ids_iter:
                generated_ids = generated_ids.tolist()
                if len(generated_ids) == 0:
                    break
                semantic_token_ids_arr.extend(generated_ids)
        except BaseException as exc:
            llm_error[0] = f'{type(exc).__name__}: {exc}'
        finally:
            llm_is_done_flag[0] = True
"""
    text = replace_once(text, old, new, 'worker error propagation')
    text = replace_once(text,
        "                llm_is_done_flag = [False]\n",
        "                llm_is_done_flag = [False]\n                llm_error = [None]\n",
        'worker error state')
    text = replace_once(text,
        '                    args=(generated_ids_iter, semantic_token_ids_arr, llm_is_done_flag)\n',
        '                    args=(generated_ids_iter, semantic_token_ids_arr, llm_is_done_flag, llm_error)\n',
        'worker error argument')
    text = replace_once(text,
        '                    if llm_is_done_flag[0]:\n                        break\n',
        "                    if llm_is_done_flag[0]:\n                        if llm_error[0]:\n                            raise pb_utils.TritonModelException(llm_error[0])\n                        break\n",
        'check worker error')
    text = replace_once(text,
        '                this_tts_speech_token = torch.tensor(semantic_token_ids_arr).unsqueeze(dim=0).to(torch.int32).to(self.device)\n',
        "                if not semantic_token_ids_arr:\n                    raise pb_utils.TritonModelException('TRT returned no speech tokens')\n                this_tts_speech_token = torch.tensor(semantic_token_ids_arr).unsqueeze(dim=0).to(torch.int32).to(self.device)\n",
        'reject empty speech tokens')

    # Capture the exact TRT-generated speech tokens from each completed request.
    # This does not change the token sequence, chunk boundaries, or inference.
    capture_old = """                this_tts_speech_token = torch.tensor(semantic_token_ids_arr).unsqueeze(dim=0).to(torch.int32).to(self.device)
                sub_tts_speech = self.forward_token2wav(this_tts_speech_token, token2wav_request_id, prompt_speech_tokens, prompt_speech_feat, prompt_spk_embedding, token_offset, True)
"""
    capture_new = """                _capture_dir = os.environ.get('NAVOIY_TOKEN_CAPTURE_DIR')
                if _capture_dir:
                    os.makedirs(_capture_dir, exist_ok=True)
                    _capture_id = str(token2wav_request_id)
                    if not __import__('re').fullmatch(r'[0-9a-fA-F-]{36}', _capture_id):
                        raise pb_utils.TritonModelException('invalid capture request identifier')
                    _ids = np.asarray(semantic_token_ids_arr, dtype=np.int32)
                    np.save(os.path.join(_capture_dir, _capture_id + '.npy'), _ids, allow_pickle=False)
                    _capture_meta = {'request_id': _capture_id, 'normalized_text': target_text,
                                     'speech_token_count': int(_ids.size),
                                     'speech_token_dtype': 'int32', 'sampling_seed': 42}
                    with open(os.path.join(_capture_dir, _capture_id + '.json'), 'w', encoding='utf-8') as _f:
                        json.dump(_capture_meta, _f, ensure_ascii=False, indent=2)
                    self.logger.log_info(f'NAVOIY_TOKEN_CAPTURE request={_capture_id} count={_ids.size}')
                this_tts_speech_token = torch.tensor(semantic_token_ids_arr).unsqueeze(dim=0).to(torch.int32).to(self.device)
                sub_tts_speech = self.forward_token2wav(this_tts_speech_token, token2wav_request_id, prompt_speech_tokens, prompt_speech_feat, prompt_spk_embedding, token_offset, True)
                # Optional fallback mode: smooth ONLY the terminal waveform boundary.
                # Default mode is OFF; no tokens or preceding streaming chunks change.
                if os.environ.get('NAVOIY_ENDING_MODE', 'original') == 'soften':
                    _fade_ms = float(os.environ.get('NAVOIY_FINAL_FADE_MS', '15'))
                    _silence_ms = float(os.environ.get('NAVOIY_FINAL_SILENCE_MS', '200'))
                    _fade_samples = min(int(_fade_ms * 24), int(sub_tts_speech.shape[-1]))
                    _silence_samples = max(0, int(_silence_ms * 24))
                    if _fade_samples > 0:
                        sub_tts_speech = sub_tts_speech.clone()
                        _ramp = torch.linspace(1., 0., _fade_samples, device=sub_tts_speech.device,
                                               dtype=sub_tts_speech.dtype)
                        sub_tts_speech[..., -_fade_samples:] *= _ramp
                    if _silence_samples > 0:
                        _pad = torch.zeros(*sub_tts_speech.shape[:-1], _silence_samples,
                                           device=sub_tts_speech.device, dtype=sub_tts_speech.dtype)
                        sub_tts_speech = torch.cat([sub_tts_speech, _pad], dim=-1)
                    self.logger.log_info(f'NAVOIY_TAIL_SOFTEN request={_capture_dir or "disabled"} fade_ms={_fade_ms} silence_ms={_silence_ms}')
"""
    text = replace_once(text, capture_old, capture_new, 'capture tokens and optional final-tail soften')

    if '"temperature": np.array([[0.8]], dtype=np.float32),' not in text:
        raise RuntimeError("Could not verify upstream sampling block")

    path.write_text(text, encoding="utf-8")
    print(f"Patched Navoiy instruct2 orchestration: {path}", flush=True)


if __name__ == "__main__":
    main()
