import torch

from megablocks.layers import common
from megablocks.layers.arguments import Arguments


# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign tokens uniformly
# across the experts. We do this with a custom autograd operation
# so that PyTorch still executes the full set of router operation.
class _UniformExpertAssignment(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, num_experts):
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)


_uniform_expert_assignment = _UniformExpertAssignment.apply


class LearnedRouter(torch.nn.Module):

    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args

        # Learned router parameters.
        #
        # NOTE: This weight matrix is not parallelized with expert model
        # parallelism. Each device needs the entire router weight matrix
        # so that it can route its batch of data correctly.
        self.layer = torch.nn.Linear(
            args.hidden_size,
            args.moe_num_experts,
            bias=False,
            dtype=common.dtype(args),
            device=args.device,
        )
        args.init_method(self.layer.weight)

        if args.moe_router_type == "word":
            print("Using word-level routing.")
            self.forward = self.forward_word
        elif args.moe_router_type == "sentence":
            print("Using sentence-level routing.")
            self.forward = self.forward_sentence

    def jitter(self, x):
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    def forward(
        self,
        x,
        random_router: bool = False,
        prune_list: torch.Tensor | None = None,
        word_ids: torch.Tensor | None = None,
        sentence_ids: torch.Tensor | None = None,
    ):
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        if self.args.moe_expert_choice:
            # Get probability for each token
            bs, sq, _ = x.shape
            capacity = (
                self.args.moe_top_k
            )  # Use top k as the capacity to match regular MoEs
            logits = self.layer(x)
            if random_router:
                logits = torch.randn_like(logits)
            if prune_list is not None:
                prune_list = prune_list.to(logits.device)
                logits[..., prune_list] = float("-inf")
            scores = logits.softmax(dim=-1)  # [batch_size, seq_len, num_experts]
            expert_weights, expert_indices = torch.topk(
                scores.transpose(1, 2),
                (capacity * sq) // self.args.moe_num_experts,
                dim=-1,
            )  # [batch_size, num_experts, k]
        elif self.args.moe_expert_choice_grouped:
            bs, sq, _ = x.shape
            capacity = (
                self.args.moe_top_k
            )  # Use top k as the capacity to match regular MoEs
            logits = self.layer(x.view(-1, x.shape[-1]))  # [bs & sq, num_experts]
            if random_router:
                logits = torch.randn_like(logits)
            if prune_list is not None:
                prune_list = prune_list.to(logits.device)
                logits[..., prune_list] = float("-inf")
            scores = logits.softmax(dim=-1)
            expert_weights, expert_indices = torch.topk(
                scores.transpose(0, 1),
                (capacity * bs * sq) // self.args.moe_num_experts,
                dim=-1,
            )  # [num_experts, k]
        else:
            logits = self.layer(x.view(-1, x.shape[-1]))
            if random_router:
                logits = torch.randn_like(logits)
            if prune_list is not None:
                prune_list = prune_list.to(logits.device)
                logits[..., prune_list] = float("-inf")
            scores = logits.softmax(dim=-1)
            expert_weights, expert_indices = self._top_k(scores)

        if self.args.moe_normalize_expert_weights:
            expert_weights = expert_weights / torch.norm(
                expert_weights,
                p=self.args.moe_normalize_expert_weights,
                dim=-1,
                keepdim=True,
            )

        expert_indices = (
            _uniform_expert_assignment(expert_indices, self.args.moe_num_experts)
            if self.args.uniform_expert_assignment
            else expert_indices
        )
        return scores, logits, expert_weights, expert_indices

    def forward_word(
        self,
        x,
        random_router: bool = False,
        prune_list: torch.Tensor | None = None,
        word_ids: torch.Tensor | None = None,
        sentence_ids: torch.Tensor | None = None,
    ):
        """
        Word-wise routing version for MoWE.
        This implementation computes routing on a per-word basis and then
        “expands” the results back to token-level such that the return values
        have the same shape as the token-wise router.
        
        Assumptions:
         - x has shape [batch_size, seq_len, hidden_size]
         - word_ids has shape [batch_size, seq_len] where tokens belonging to the same word share the same ID.
        """
        if word_ids is None:
            raise ValueError("word_ids must be provided for word-wise routing.")

        # Optionally add jitter.
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        batch, seq_len, hidden_size = x.shape

        # Lists to hold per-sample word representations and a mapping from token index to word index.
        word_reps_list = []       # Each entry: [num_words_i, hidden_size]
        token_to_word_list = []   # Each entry: [seq_len] mapping tokens to word index
        valid_counts = []         # Number of words per sample

        for i in range(batch):
            # For each sample, compute a mask for the first token of each word.
            sample_word_ids = word_ids[i]  # shape: [seq_len]
            # Mark first token as true.
            first_token_mask = torch.ones_like(sample_word_ids, dtype=torch.bool)
            if seq_len > 1:
                first_token_mask[1:] = sample_word_ids[1:] != sample_word_ids[:-1]
            # Compute cumulative sum to get word indices for each token.
            # For example, if first_token_mask is [1, 0, 1, 0, 0, 1], then cumsum is [1,1,2,2,2,3].
            # Subtract 1 to make word indices start at 0.
            token_to_word = torch.cumsum(first_token_mask.to(torch.int64), dim=0) - 1
            token_to_word_list.append(token_to_word)
            # Extract the representative word embeddings (first token of each word).
            word_rep = x[i][first_token_mask]  # shape: [num_words_i, hidden_size]
            word_reps_list.append(word_rep)
            valid_counts.append(word_rep.size(0))

        # Determine maximum number of words in the batch to pad to a fixed shape.
        max_words = max(valid_counts)
        padded_word_reps = []
        for word_rep in word_reps_list:
            num_words = word_rep.size(0)
            if num_words < max_words:
                pad_tensor = torch.zeros((max_words - num_words, hidden_size),
                                         dtype=word_rep.dtype, device=word_rep.device)
                padded = torch.cat([word_rep, pad_tensor], dim=0)
            else:
                padded = word_rep
            padded_word_reps.append(padded)
        # Stack to shape: [batch, max_words, hidden_size]
        word_reps_batch = torch.stack(padded_word_reps, dim=0)

        # Compute word-level logits.
        logits_word = self.layer(word_reps_batch)  # [batch, max_words, moe_num_experts]
        if random_router:
            logits_word = torch.randn_like(logits_word)
        if prune_list is not None:
            prune_list = prune_list.to(logits_word.device)
            logits_word[..., prune_list] = float("-inf")
        # Compute routing scores.
        scores_word = logits_word.softmax(dim=-1)  # [batch, max_words, moe_num_experts]

        # Compute top-k expert selection on word-level scores.
        expert_weights_word, expert_indices_word = self._top_k(scores_word)
        if self.args.moe_normalize_expert_weights:
            expert_weights_word = expert_weights_word / torch.norm(
                expert_weights_word,
                p=self.args.moe_normalize_expert_weights,
                dim=-1,
                keepdim=True,
            )
        if self.args.uniform_expert_assignment:
            expert_indices_word = _uniform_expert_assignment(expert_indices_word, self.args.moe_num_experts)

        # Now, for each sample, expand the word-level routing outputs back to token-level.
        # We use the previously computed token_to_word mapping.
        scores_token = []
        logits_token = []
        expert_weights_token = []
        expert_indices_token = []

        for i in range(batch):
            # Get mapping for this sample: shape [seq_len]
            mapping = token_to_word_list[i]  # LongTensor of shape [seq_len]
            # mapping values are in [0, num_words_i). They must be used to index into the i-th row of padded outputs.
            # Use torch.gather to map word-level outputs to tokens.
            # First, expand mapping to have a last dimension of 1.
            mapping_expanded = mapping.unsqueeze(-1)  # [seq_len, 1]

            # For scores and logits, which have shape [num_words, moe_num_experts].
            s_token = torch.gather(scores_word[i], 0, mapping_expanded.expand(-1, scores_word.size(-1)))
            l_token = torch.gather(logits_word[i], 0, mapping_expanded.expand(-1, logits_word.size(-1)))
            # For expert_weights and indices, which have shape [num_words, moe_top_k].
            ew_token = torch.gather(expert_weights_word[i], 0, mapping_expanded.expand(-1, expert_weights_word.size(-1)))
            ei_token = torch.gather(expert_indices_word[i], 0, mapping_expanded.expand(-1, expert_indices_word.size(-1)))

            scores_token.append(s_token)
            logits_token.append(l_token)
            expert_weights_token.append(ew_token)
            expert_indices_token.append(ei_token)

        # Stack token-level outputs to have shape [batch, seq_len, ...]
        scores_token = torch.stack(scores_token, dim=0)
        logits_token = torch.stack(logits_token, dim=0)
        expert_weights_token = torch.stack(expert_weights_token, dim=0)
        expert_indices_token = torch.stack(expert_indices_token, dim=0)

        # Return values matching token-wise router: shapes [batch, seq_len, moe_num_experts] and
        # for top-k outputs, [batch, seq_len, moe_top_k].
        return scores_token, logits_token, expert_weights_token, expert_indices_token
    
    def forward_sentence(
        self,
        x,
        random_router: bool = False,
        prune_list: torch.Tensor | None = None,
        word_ids: torch.Tensor | None = None,
        sentence_ids: torch.Tensor | None = None,
    ):
        """
        Sentence-level routing version for MoWE.
        This implementation performs the routing on a per-sentence basis and then
        expands the results back to token-level so that the final outputs have the same
        shapes as in the token-wise router.
        
        Assumptions:
        - x has shape [batch_size, seq_len, hidden_size]
        - sentence_ids has shape [batch_size, seq_len] with contiguous tokens 
            sharing the same sentence ID.
        """
        if sentence_ids is None:
            raise ValueError("sentence_ids must be provided for sentence-wise routing.")

        # Optionally add jitter.
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        batch, seq_len, hidden_size = x.shape

        # We'll compute, for each sample:
        # 1. A boolean mask (first token of each sentence).
        # 2. A mapping from token position to sentence index.
        # 3. The representative sentence embeddings.
        sentence_reps_list = []      # List of tensors of shape [num_sentences_i, hidden_size]
        token_to_sentence_list = []  # List of tensors of shape [seq_len]
        valid_counts = []            # Number of sentences per sample

        for i in range(batch):
            sample_sentence_ids = sentence_ids[i]  # shape: [seq_len]
            # Create a boolean mask that is True at the start of each sentence.
            first_token_mask = torch.ones_like(sample_sentence_ids, dtype=torch.bool)
            if seq_len > 1:
                first_token_mask[1:] = sample_sentence_ids[1:] != sample_sentence_ids[:-1]
            # Compute token-to-sentence mapping using cumulative sum.
            token_to_sentence = torch.cumsum(first_token_mask.to(torch.int64), dim=0) - 1
            token_to_sentence_list.append(token_to_sentence)
            # Extract sentence representations using the mask:
            sentence_rep = x[i][first_token_mask]  # shape: [num_sentences_i, hidden_size]
            sentence_reps_list.append(sentence_rep)
            valid_counts.append(sentence_rep.size(0))

        # Pad sentence representations to have uniform shape across the batch.
        max_sentences = max(valid_counts)
        padded_sentence_reps = []
        for sentence_rep in sentence_reps_list:
            num_sentences = sentence_rep.size(0)
            if num_sentences < max_sentences:
                pad_tensor = torch.zeros((max_sentences - num_sentences, hidden_size),
                                        dtype=sentence_rep.dtype, device=sentence_rep.device)
                padded = torch.cat([sentence_rep, pad_tensor], dim=0)
            else:
                padded = sentence_rep
            padded_sentence_reps.append(padded)
        # Form a batch tensor of shape: [batch, max_sentences, hidden_size]
        sentence_reps_batch = torch.stack(padded_sentence_reps, dim=0)

        # Compute sentence-level logits using the learned linear router.
        logits_sentence = self.layer(sentence_reps_batch)  # [batch, max_sentences, moe_num_experts]
        if random_router:
            logits_sentence = torch.randn_like(logits_sentence)
        if prune_list is not None:
            prune_list = prune_list.to(logits_sentence.device)
            logits_sentence[..., prune_list] = float("-inf")
        # Compute routing probabilities.
        scores_sentence = logits_sentence.softmax(dim=-1)  # [batch, max_sentences, moe_num_experts]

        # Compute top-k expert assignment at the sentence level.
        expert_weights_sentence, expert_indices_sentence = self._top_k(scores_sentence)
        if self.args.moe_normalize_expert_weights:
            expert_weights_sentence = expert_weights_sentence / torch.norm(
                expert_weights_sentence,
                p=self.args.moe_normalize_expert_weights,
                dim=-1,
                keepdim=True,
            )
        if self.args.uniform_expert_assignment:
            expert_indices_sentence = _uniform_expert_assignment(expert_indices_sentence, self.args.moe_num_experts)

        # Expand sentence-level outputs back to token-level using the token-to-sentence mapping.
        scores_token = []
        logits_token = []
        expert_weights_token = []
        expert_indices_token = []

        for i in range(batch):
            mapping = token_to_sentence_list[i]  # Shape: [seq_len]
            # Expand mapping to shape [seq_len, 1] for gathering.
            mapping_expanded = mapping.unsqueeze(-1)
            # Gather token-level outputs from sentence-level tensors.
            s_token = torch.gather(scores_sentence[i], 0, mapping_expanded.expand(-1, scores_sentence.size(-1)))
            l_token = torch.gather(logits_sentence[i], 0, mapping_expanded.expand(-1, logits_sentence.size(-1)))
            ew_token = torch.gather(expert_weights_sentence[i], 0, mapping_expanded.expand(-1, expert_weights_sentence.size(-1)))
            ei_token = torch.gather(expert_indices_sentence[i], 0, mapping_expanded.expand(-1, expert_indices_sentence.size(-1)))

            scores_token.append(s_token)
            logits_token.append(l_token)
            expert_weights_token.append(ew_token)
            expert_indices_token.append(ei_token)

        # Stack token-level outputs to get tensors of shape [batch, seq_len, ...]
        scores_token = torch.stack(scores_token, dim=0)
        logits_token = torch.stack(logits_token, dim=0)
        expert_weights_token = torch.stack(expert_weights_token, dim=0)
        expert_indices_token = torch.stack(expert_indices_token, dim=0)

        # Return values in the same format as the token-wise router:
        # scores: [batch, seq_len, moe_num_experts]
        # expert_weights & expert_indices: [batch, seq_len, moe_top_k]
        return scores_token, logits_token, expert_weights_token, expert_indices_token