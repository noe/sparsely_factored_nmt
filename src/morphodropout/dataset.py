import torch
from itertools import accumulate
from fairseq.data import (
    data_utils,
    FairseqDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
)
from functools import lru_cache
import numpy as np
from seqp.hdf5 import Hdf5RecordReader
from typing import List

from morphodropout.binarize import (
    SRC_SUBWORD_KEY,
    SRC_SUBWORD_LENGTHS_KEY,
    SRC_MORPH_KEY,
    SRC_MORPH_LENGTHS_KEY,
    SRC_LEMMA_KEY,
)


class MorphoDataset(FairseqDataset):
    """A dataset that provides helpers for batching."""

    def __init__(self,
                 dictionary: Dictionary,
                 data_files: List[str],
                 morpho_dropout: float = 0.5):
        self.dictionary = dictionary
        self.pad_idx = dictionary.pad_index
        self.data_files = data_files
        self.reader = Hdf5RecordReader(data_files)
        self.morpho_dropout = morpho_dropout

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle the reader (see https://github.com/h5py/h5py/issues/1092)
        del state["reader"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add reader back since it doesn't exist in the pickle
        self.reader = Hdf5RecordReader(self.data_files)

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        record = self.reader.retrieve(index)
        lemmas = record[SRC_LEMMA_KEY]
        subwords = record[SRC_SUBWORD_KEY]
        morphos = record[SRC_MORPH_KEY]
        morpho_lengths = record[SRC_MORPH_LENGTHS_KEY].tolist()
        sw_lengths = record[SRC_SUBWORD_LENGTHS_KEY].tolist()

        use_subwords = np.random.binomial(size=len(lemmas), n=1, p=self.morpho_dropout).astype(bool)
        EOS_POS = -1
        sw_pos = list(accumulate(sw_lengths))
        morpho_pos = list(accumulate(morpho_lengths))
        start_end = zip([0] + sw_pos[:EOS_POS - 1],
                        sw_pos[:EOS_POS],
                        [0] + morpho_pos[:-1],
                        morpho_pos)

        pad = [self.dictionary.pad_index]
        eos = [self.dictionary.eos_index]

        result: List[np.ndarray] = list()
        max_depth = 0
        for k, (sw_start, sw_end, morpho_start, morpho_end) in enumerate(start_end):
            if use_subwords[k] or lemmas[k] in [self.dictionary.unk_index, self.dictionary.pad_index]:
                word_subwords = subwords[sw_start:sw_end]
                result.extend([np.array([sw], dtype=np.int64) for sw in word_subwords])
                max_depth = max(max_depth, 1)
            else:
                clean_morphos = [m if m != self.dictionary.unk_index else self.dictionary.pad_index
                                 for m in morphos[morpho_start:morpho_end]]
                word_linguistic_info = [lemmas[k]] + clean_morphos
                result.append(np.array(word_linguistic_info, dtype=np.int64))
                max_depth = max(max_depth, len(word_linguistic_info))
        result.append(np.array(eos, dtype=np.int64))

        # Add padding and convert to tensors
        result = [torch.LongTensor(np.concatenate((r, pad * (max_depth - len(r))))) for r in result]

        # Combine padded tensors into a single 2d tensor
        result = torch.stack(result)

        return result

    def __len__(self):
        return self.reader.num_records()

    def num_tokens(self, index):
        return self.reader.length(index)

    def size(self, index):
        return self.reader.length(index)

    def ordered_indices(self):
        return [idx for idx, length in self.reader.indexes_and_lengths()]

    @property
    def sizes(self):
        return [length for idx, length in self.reader.indexes_and_lengths()]

    def get_dummy_batch(self, num_tokens, max_positions):
        return self.dictionary.dummy_sentence(num_tokens).unsqueeze(dim=2)

    @property
    def supports_prefetch(self):
        return False

    def prefetch(self, indices):
        raise NotImplementedError

    def collater(self, samples):
        size = max(v.size(0) for v in samples)
        depth = max(v.size(1) for v in samples)
        res = samples[0].new(len(samples), size, depth).fill_(self.pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(samples):
            copy_tensor(v, res[i, :v.size(0), :v.size(1)])
        return res


class MonolingualDataset(FairseqDataset):
    """A dataset that provides helpers for batching."""

    def __init__(self,
                 dictionary: Dictionary,
                 data_files: List[str],
                 left_pad=False,
                 move_eos_to_beginning=False):
        self.dictionary = dictionary
        self.data_files = data_files
        self.reader = Hdf5RecordReader(data_files)
        self.left_pad = left_pad
        self.move_eos_to_beginning = move_eos_to_beginning

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle the reader (see https://github.com/h5py/h5py/issues/1092)
        del state["reader"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add reader back since it doesn't exist in the pickle
        self.reader = Hdf5RecordReader(self.data_files)

    def __getitem__(self, index):
        elem = self.reader.retrieve(index)
        elem = torch.LongTensor(elem.astype(np.int64))
        return elem

    def __len__(self):
        return self.reader.num_records()

    def collater(self, samples):
        tokens = data_utils.collate_tokens(
                        [s for s in samples],
                        self.dictionary.pad_index,
                        self.dictionary.eos_index,
                        self.left_pad,
                        move_eos_to_beginning=self.move_eos_to_beginning)
        return tokens

    def get_dummy_batch(self, num_tokens, max_positions):
        return self.dictionary.dummy_sentence(num_tokens)

    def num_tokens(self, index):
        return self.reader.length(index)

    def size(self, index):
        return self.reader.length(index)

    def ordered_indices(self):
        return [idx for idx, length in self.reader.indexes_and_lengths()]

    @property
    def sizes(self):
        return [length for idx, length in self.reader.indexes_and_lengths()]

    @property
    def supports_prefetch(self):
        return False

    def prefetch(self, indices):
        raise NotImplementedError


def build_combined_dataset(
        src_dictionary: Dictionary,
        src_data_files: List[str],
        src_morpho_dropout: float,
        tgt_dictionary: Dictionary,
        tgt_hdf5_files: List[str],
        seed: int,
        epoch: int,
        ) -> FairseqDataset:

    src = MorphoDataset(src_dictionary,
                        src_data_files,
                        src_morpho_dropout)

    target = MonolingualDataset(tgt_dictionary,
                                tgt_hdf5_files,
                                move_eos_to_beginning=False)

    prev_output_tokens = MonolingualDataset(tgt_dictionary,
                                            tgt_hdf5_files,
                                            move_eos_to_beginning=True)

    return NestedDictionaryDataset(
                {
                    'id': IdDataset(),
                    'net_input': {
                        'src_tokens': src,
                        'src_lengths': NumelDataset(src, reduce=False),
                        'prev_output_tokens': prev_output_tokens,
                    },
                    'target': target,
                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(src, reduce=True),
                },
                sizes=[src.sizes],
            )
