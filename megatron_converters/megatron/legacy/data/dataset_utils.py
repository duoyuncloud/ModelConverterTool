# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, and NVIDIA.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Most of the code here has been copied from:
#   https://github.com/google-research/albert/blob/master/create_pretraining_data.py
# with some modifications.

import math
import os
import time
import collections

import numpy as np
import torch

from megatron.training import get_args, print_rank_0
from megatron.core import mpu
from megatron.core.datasets.indexed_dataset import IndexedDataset


DSET_TYPE_BERT = "standard_bert"
DSET_TYPE_ICT = "ict"
DSET_TYPE_T5 = "t5"
DSET_TYPE_MULTIMODAL = "multimodal"

DSET_TYPES = [DSET_TYPE_BERT, DSET_TYPE_ICT, DSET_TYPE_T5, DSET_TYPE_MULTIMODAL]


def get_datasets_weights_and_num_samples(data_prefix, train_valid_test_num_samples):

    # The data prefix should be in the format of:
    #   weight-1, data-prefix-1, weight-2, data-prefix-2, ..
    assert len(data_prefix) % 2 == 0
    num_datasets = len(data_prefix) // 2
    weights = [0] * num_datasets
    prefixes = [0] * num_datasets
    for i in range(num_datasets):
        weights[i] = float(data_prefix[2 * i])
        prefixes[i] = (data_prefix[2 * i + 1]).strip()
    # Normalize weights
    weight_sum = 0.0
    for weight in weights:
        weight_sum += weight
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]

    # Add 0.5% (the 1.005 factor) so in case the bleding dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    if isinstance(train_valid_test_num_samples, list):
        datasets_train_valid_test_num_samples = []
        for weight in weights:
            datasets_train_valid_test_num_samples.append(
                [int(math.ceil(val * weight * 1.005)) for val in train_valid_test_num_samples]
            )
    else:
        # Used when separate dataset files are provided for train,
        # valid and test
        datasets_train_valid_test_num_samples = [
            int(math.ceil(train_valid_test_num_samples * weight * 1.005)) for weight in weights
        ]

    return prefixes, weights, datasets_train_valid_test_num_samples


def get_a_and_b_segments(sample, np_rng):
    """Divide sample into a and b segments."""

    # Number of sentences in the sample.
    n_sentences = len(sample)
    # Make sure we always have two sentences.
    assert n_sentences > 1, "make sure each sample has at least two sentences."

    # First part:
    # `a_end` is how many sentences go into the `A`.
    a_end = 1
    if n_sentences >= 3:
        # Note that randin in numpy is exclusive.
        a_end = np_rng.randint(1, n_sentences)
    tokens_a = []
    for j in range(a_end):
        tokens_a.extend(sample[j])

    # Second part:
    tokens_b = []
    for j in range(a_end, n_sentences):
        tokens_b.extend(sample[j])

    # Random next:
    is_next_random = False
    if np_rng.random() < 0.5:
        is_next_random = True
        tokens_a, tokens_b = tokens_b, tokens_a

    return tokens_a, tokens_b, is_next_random


def truncate_segments(tokens_a, tokens_b, len_a, len_b, max_num_tokens, np_rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    # print(len_a, len_b, max_num_tokens)
    assert len_a > 0
    if len_a + len_b <= max_num_tokens:
        return False
    while len_a + len_b > max_num_tokens:
        if len_a > len_b:
            len_a -= 1
            tokens = tokens_a
        else:
            len_b -= 1
            tokens = tokens_b
        if np_rng.random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()
    return True


def create_tokens_and_tokentypes(tokens_a, tokens_b, cls_id, sep_id):
    """Merge segments A and B, add [CLS] and [SEP] and build tokentypes."""

    tokens = []
    tokentypes = []
    # [CLS].
    tokens.append(cls_id)
    tokentypes.append(0)
    # Segment A.
    for token in tokens_a:
        tokens.append(token)
        tokentypes.append(0)
    # [SEP].
    tokens.append(sep_id)
    tokentypes.append(0)
    # Segment B.
    for token in tokens_b:
        tokens.append(token)
        tokentypes.append(1)
    if tokens_b:
        # [SEP].
        tokens.append(sep_id)
        tokentypes.append(1)

    return tokens, tokentypes


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def is_start_piece(piece):
    """Check if the current word piece is the starting piece (BERT)."""
    # When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    return not piece.startswith("##")


def create_masked_lm_predictions(
    tokens,
    vocab_id_list,
    vocab_id_to_token_dict,
    masked_lm_prob,
    cls_id,
    sep_id,
    mask_id,
    max_predictions_per_seq,
    np_rng,
    max_ngrams=3,
    do_whole_word_mask=True,
    favor_longer_ngram=False,
    do_permutation=False,
    geometric_dist=False,
    masking_style="bert",
):
    """Creates the predictions for the masked LM objective.
    Note: Tokens here are vocab ids and not text tokens."""

    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    token_boundary = [0] * len(tokens)

    for i, token in enumerate(tokens):
        if token == cls_id or token == sep_id:
            token_boundary[i] = 1
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if do_whole_word_mask and len(cand_indexes) >= 1 and not is_start_piece(vocab_id_to_token_dict[token]):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if is_start_piece(vocab_id_to_token_dict[token]):
                token_boundary[i] = 1

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary)

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    ngrams = np.arange(1, max_ngrams + 1, dtype=np.int64)
    if not geometric_dist:
        # Note(mingdachen):
        # By default, we set the probilities to favor shorter ngram sequences.
        pvals = 1.0 / np.arange(1, max_ngrams + 1)
        pvals /= pvals.sum(keepdims=True)
        if favor_longer_ngram:
            pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx : idx + n])
        ngram_indexes.append(ngram_index)

    np_rng.shuffle(ngram_indexes)

    (masked_lms, masked_spans) = ([], [])
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        if not geometric_dist:
            n = np_rng.choice(
                ngrams[: len(cand_index_set)],
                p=pvals[: len(cand_index_set)] / pvals[: len(cand_index_set)].sum(keepdims=True),
            )
        else:
            # Sampling "n" from the geometric distribution and clipping it to
            # the max_ngrams. Using p=0.2 default from the SpanBERT paper
            # https://arxiv.org/pdf/1907.10529.pdf (Sec 3.1)
            n = min(np_rng.geometric(0.2), max_ngrams)

        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            if masking_style == "bert":
                # 80% of the time, replace with [MASK]
                if np_rng.random() < 0.8:
                    masked_token = mask_id
                else:
                    # 10% of the time, keep original
                    if np_rng.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = vocab_id_list[np_rng.randint(0, len(vocab_id_list))]
            elif masking_style == "t5":
                masked_token = mask_id
            else:
                raise ValueError("invalid value of masking style")

            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

        masked_spans.append(MaskedLmInstance(index=index_set, label=[tokens[index] for index in index_set]))

    assert len(masked_lms) <= num_to_predict
    np_rng.shuffle(ngram_indexes)

    select_indexes = set()
    if do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(
                ngrams[: len(cand_index_set)],
                p=pvals[: len(cand_index_set)] / pvals[: len(cand_index_set)].sum(keepdims=True),
            )
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        np_rng.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    # Sort the spans by the index of the first span
    masked_spans = sorted(masked_spans, key=lambda x: x.index[0])

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary, masked_spans)


def pad_and_convert_to_numpy(tokens, tokentypes, masked_positions, masked_labels, pad_id, max_seq_length):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(tokentypes) == num_tokens
    assert len(masked_positions) == len(masked_labels)

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64)
    tokentypes_np = np.array(tokentypes + filler, dtype=np.int64)

    # Padding mask.
    padding_mask_np = np.array([1] * num_tokens + [0] * padding_length, dtype=np.int64)

    # Lables and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)):
        assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    return tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np


def build_train_valid_test_datasets_with_prefixes(
    train_valid_test_num_samples,
    max_seq_length,
    seed,
    train_data_prefix=None,
    valid_data_prefix=None,
    test_data_prefix=None,
    binary_head=False,
    max_seq_length_dec=None,
    dataset_type="standard_bert",
):
    print_rank_0("Separate data paths provided for train, valid & test.")

    train_dataset, valid_dataset, test_dataset = None, None, None
    # Single dataset.
    if train_data_prefix is not None:
        train_dataset = build_dataset(
            "train",
            train_data_prefix,
            train_valid_test_num_samples[0],
            max_seq_length,
            seed,
            binary_head,
            max_seq_length_dec,
            dataset_type=dataset_type,
        )

    if valid_data_prefix is not None:
        valid_dataset = build_dataset(
            "valid",
            valid_data_prefix,
            train_valid_test_num_samples[1],
            max_seq_length,
            seed,
            False,
            binary_head,
            max_seq_length_dec,
            dataset_type=dataset_type,
        )

    if test_data_prefix is not None:
        test_dataset = build_dataset(
            "test",
            test_data_prefix,
            train_valid_test_num_samples[2],
            max_seq_length,
            seed,
            False,
            binary_head,
            max_seq_length_dec,
            dataset_type=dataset_type,
        )

    return (train_dataset, valid_dataset, test_dataset)


def build_train_valid_test_datasets(
    data_prefix,
    splits_string,
    train_valid_test_num_samples,
    max_seq_length,
    seed,
    binary_head=False,
    max_seq_length_dec=None,
    dataset_type="standard_bert",
):

    if len(data_prefix) == 1:
        return _build_train_valid_test_datasets(
            data_prefix[0],
            splits_string,
            train_valid_test_num_samples,
            max_seq_length,
            seed,
            binary_head,
            max_seq_length_dec,
            dataset_type=dataset_type,
        )

    raise NotImplementedError("Blending currently unsupported for non-GPT dataset instances")


def _build_train_valid_test_datasets(
    data_prefix,
    splits_string,
    train_valid_test_num_samples,
    max_seq_length,
    seed,
    binary_head,
    max_seq_length_dec,
    dataset_type="standard_bert",
):

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix, dataset_type)

    # Get start and end indices of train/valid/train into doc-idx
    # Note that doc-idx is desinged to be num-docs + 1 so we can
    # easily iterate over it.
    total_num_of_documents = indexed_dataset.document_indices.shape[0] - 1
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(" > dataset split:")

    def print_split_stats(name, index):
        print_rank_0("    {}:".format(name))
        print_rank_0(
            "     document indices in [{}, {}) total of {} "
            "documents".format(splits[index], splits[index + 1], splits[index + 1] - splits[index])
        )
        start_index = indexed_dataset.document_indices[splits[index]]
        end_index = indexed_dataset.document_indices[splits[index + 1]]
        print_rank_0(
            "     sentence indices in [{}, {}) total of {} "
            "sentences".format(start_index, end_index, end_index - start_index)
        )

    print_split_stats("train", 0)
    print_split_stats("validation", 1)
    print_split_stats("test", 2)

    def build_split_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            # Get the pointer to the original doc-idx so we can set it later.
            doc_idx_ptr = indexed_dataset.get_document_indices()
            # Slice the doc-idx
            start_index = splits[index]
            # Add +1 so we can index into the dataset to get the upper bound.
            end_index = splits[index + 1] + 1
            # New doc_idx view.
            indexed_dataset.set_document_indices(doc_idx_ptr[start_index:end_index])

            dataset = build_dataset(
                name,
                data_prefix,
                train_valid_test_num_samples[index],
                max_seq_length,
                seed,
                binary_head,
                max_seq_length_dec,
                dataset_type,
                indexed_dataset,
            )

            # Set the original pointer so dataset remains the main dataset.
            indexed_dataset.set_document_indices(doc_idx_ptr)
            # Checks.
            assert indexed_dataset.document_indices[0] == 0
            assert indexed_dataset.document_indices.shape[0] == (total_num_of_documents + 1)
        return dataset

    train_dataset = build_split_dataset(0, "train")
    valid_dataset = build_split_dataset(1, "valid")
    test_dataset = build_split_dataset(2, "test")

    return (train_dataset, valid_dataset, test_dataset)


def build_dataset(
    name,
    data_prefix,
    max_num_samples,
    max_seq_length,
    seed,
    binary_head,
    max_seq_length_dec,
    dataset_type="standard_bert",
    indexed_dataset=None,
):

    from megatron.legacy.data.ict_dataset import ICTDataset
    from megatron.legacy.data.multimodal_dataset import MultiModalDataset

    if dataset_type == DSET_TYPE_BERT or dataset_type == DSET_TYPE_T5:
        raise ValueError("The Megatron-LM BERT and T5 datasets are deprecated.")

    if dataset_type not in DSET_TYPES:
        raise ValueError("Invalid dataset_type: ", dataset_type)

    if indexed_dataset is None:
        indexed_dataset = get_indexed_dataset_(data_prefix, dataset_type)

    kwargs = dict(
        name=name,
        data_prefix=data_prefix,
        num_epochs=None,
        max_num_samples=max_num_samples,
        max_seq_length=max_seq_length,
        seed=seed,
    )

    if dataset_type == DSET_TYPE_ICT:
        args = get_args()

        title_dataset = get_indexed_dataset_(args.titles_data_path, dataset_type)

        dataset = ICTDataset(
            block_dataset=indexed_dataset,
            title_dataset=title_dataset,
            query_in_block_prob=args.query_in_block_prob,
            use_one_sent_docs=args.use_one_sent_docs,
            binary_head=binary_head,
            **kwargs,
        )
    elif dataset_type == DSET_TYPE_MULTIMODAL:
        args = get_args()
        dataset = MultiModalDataset(
            name=name,
            data_prefix=data_prefix,
            indexed_dataset=indexed_dataset,
            num_samples=max_num_samples,
            seq_length=max_seq_length,
            seed=seed,
            img_h=args.img_h,
            img_w=args.img_w,
        )
    else:
        raise NotImplementedError("Dataset type not fully implemented.")

    return dataset


def get_indexed_dataset_(data_prefix, dataset_type):

    print_rank_0(" > building dataset index ...")

    start_time = time.time()
    multimodal = dataset_type == DSET_TYPE_MULTIMODAL
    indexed_dataset = IndexedDataset(data_prefix, multimodal)
    assert indexed_dataset.sequence_lengths.shape[0] == indexed_dataset.document_indices[-1]
    print_rank_0(" > finished creating indexed dataset in {:4f} " "seconds".format(time.time() - start_time))

    print_rank_0(" > indexed dataset stats:")
    print_rank_0("    number of documents: {}".format(indexed_dataset.document_indices.shape[0] - 1))
    print_rank_0("    number of sentences: {}".format(indexed_dataset.sequence_lengths.shape[0]))

    return indexed_dataset


def get_train_valid_test_split_(splits_string, size):
    """Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(",") != -1:
        splits = [float(s) for s in splits_string.split(",")]
    elif splits_string.find("/") != -1:
        splits = [float(s) for s in splits_string.split("/")]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


def get_samples_mapping(
    indexed_dataset, data_prefix, num_epochs, max_num_samples, max_seq_length, short_seq_prob, seed, name, binary_head
):
    """Get a list that maps a sample index to a starting sentence index, end sentence index, and length"""

    if not num_epochs:
        if not max_num_samples:
            raise ValueError("Need to specify either max_num_samples " "or num_epochs")
        num_epochs = np.iinfo(np.int32).max - 1
    if not max_num_samples:
        max_num_samples = np.iinfo(np.int64).max - 1

    # Filename of the index mapping
    indexmap_filename = data_prefix
    indexmap_filename += "_{}_indexmap".format(name)
    if num_epochs != (np.iinfo(np.int32).max - 1):
        indexmap_filename += "_{}ep".format(num_epochs)
    if max_num_samples != (np.iinfo(np.int64).max - 1):
        indexmap_filename += "_{}mns".format(max_num_samples)
    indexmap_filename += "_{}msl".format(max_seq_length)
    indexmap_filename += "_{:0.2f}ssp".format(short_seq_prob)
    indexmap_filename += "_{}s".format(seed)
    indexmap_filename += ".npy"

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0 and not os.path.isfile(indexmap_filename):
        print(
            " > WARNING: could not find index map file {}, building "
            "the indices on rank 0 ...".format(indexmap_filename)
        )

        # Make sure the types match the helpers input types.
        assert indexed_dataset.document_indices.dtype == np.int64
        assert indexed_dataset.sequence_lengths.dtype == np.int32

        # Build samples mapping
        verbose = torch.distributed.get_rank() == 0
        start_time = time.time()
        print_rank_0(" > building samples index mapping for {} ...".format(name))
        # First compile and then import.
        from megatron.core.datasets import helpers

        samples_mapping = helpers.build_mapping(
            indexed_dataset.document_indices,
            indexed_dataset.sequence_lengths,
            num_epochs,
            max_num_samples,
            max_seq_length,
            short_seq_prob,
            seed,
            verbose,
            2 if binary_head else 1,
        )
        print_rank_0(" > done building samples index maping")
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
        print_rank_0(" > saved the index mapping in {}".format(indexmap_filename))
        # Make sure all the ranks have built the mapping
        print_rank_0(
            " > elasped time to build and save samples mapping " "(seconds): {:4f}".format(time.time() - start_time)
        )
    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.tensor([1], dtype=torch.long, device="cuda")
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size()
        // torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group())
    )

    # Load indexed dataset.
    print_rank_0(" > loading indexed mapping from {}".format(indexmap_filename))
    start_time = time.time()
    samples_mapping = np.load(indexmap_filename, allow_pickle=True, mmap_mode="r")
    print_rank_0("    loaded indexed file in {:3.3f} seconds".format(time.time() - start_time))
    print_rank_0("    total number of samples: {}".format(samples_mapping.shape[0]))

    return samples_mapping
