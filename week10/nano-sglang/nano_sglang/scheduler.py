"""Part 3: Continuous Batching Scheduler

Prefill each request one at a time, then batch all decodes together
using engine.decode_batch() for GPU efficiency.
"""

from .sampling import SamplingParams
from .sequence import Sequence, SequenceStatus
from .engine import Engine


class Scheduler:
    def __init__(self, model_path: str, max_batch_size: int = 64, device: str = "cuda"):
        self.engine = Engine(model_path, device=device)
        self.tokenizer = self.engine.tokenizer
        self.max_batch_size = max_batch_size

        self.next_seq_id = 0
        self.waiting_queue: list[Sequence] = []
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []

    def add_request(self, prompt: str, sampling_params: SamplingParams = None):
        """Tokenize prompt, create Sequence, add to waiting queue."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        token_ids = self.tokenizer.encode(prompt)
        seq = Sequence(
            seq_id=self.next_seq_id,
            prompt_token_ids=token_ids,
            max_tokens=sampling_params.max_tokens,
        )
        self.waiting_queue.append(seq)
        self.next_seq_id += 1

    def _prefill_waiting(self, sampling_params: SamplingParams):
        """Prefill one request from the waiting queue and move it to running."""
        if not self.waiting_queue:
            return
        if len(self.running) >= self.max_batch_size:
            return
        seq = self.waiting_queue.pop(0)
        # Apply the run-time sampling budget to requests queued earlier.
        seq.max_tokens = sampling_params.max_tokens
        first_token = self.engine.prefill(seq, sampling_params)
        seq.output_token_ids.append(first_token)
        reached_eos = first_token == self.tokenizer.eos_token_id
        reached_limit = seq.num_generated >= seq.max_tokens

        if reached_eos or reached_limit:
            seq.status = SequenceStatus.FINISHED
            self.finished.append(seq)
        else:
            self.running.append(seq)

    def _decode_running(self, sampling_params: SamplingParams):
        """Decode all running sequences in one batched forward pass."""
        if not self.running:
            return

        next_tokens = self.engine.decode_batch(self.running, sampling_params)
        still_running = []

        for seq, next_token in zip(self.running, next_tokens):
            seq.output_token_ids.append(next_token)

            reached_eos = next_token == self.tokenizer.eos_token_id
            reached_limit = seq.num_generated >= seq.max_tokens

            if reached_eos or reached_limit:
                seq.status = SequenceStatus.FINISHED
                self.finished.append(seq)
            else:
                seq.status = SequenceStatus.DECODING
                still_running.append(seq)

        # Keep only active sequences for the next scheduler step.
        self.running = still_running

    def step(self, sampling_params: SamplingParams = None):
        """One scheduling iteration."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        if self.waiting_queue:
            self._prefill_waiting(sampling_params)

        if self.running:
            self._decode_running(sampling_params)

    def run_to_completion(self, sampling_params: SamplingParams = None) -> list[str]:
        """Run all requests to completion, return generated texts in order."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        while self.waiting_queue or self.running:
            self.step(sampling_params)

        # Return outputs in request order instead of finish order.
        ordered = sorted(self.finished, key=lambda seq: seq.seq_id)
        return [self.tokenizer.decode(seq.output_token_ids) for seq in ordered]
