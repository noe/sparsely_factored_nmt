
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq import utils, search
from glob import glob
import os

from morphodropout.binarize import SRC_SIDE, TGT_SIDE
from morphodropout.dataset import build_combined_dataset
from morphodropout.seq_gen import SequenceGenerator


@register_task('morpho_translation')
class MorphoTranslation(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.morpho_dropout_final = args.morpho_dropout
        self.morpho_dropout_initial = args.morpho_dropout_initial
        self.morpho_dropout_end_epoch = args.morpho_dropout_end_epoch

    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        parser.add_argument('--morpho-dropout', type=float, default=0.5)
        parser.add_argument('--morpho-dropout-initial', type=float, default=None)
        parser.add_argument('--morpho-dropout-end-epoch', type=int, default=None)

    def morpho_dropout_for(self, epoch: int) -> float:
        if self.morpho_dropout_initial is None:
            return self.morpho_dropout_final

        assert self.morpho_dropout_end_epoch is not None

        initial = self.morpho_dropout_initial
        final = self.morpho_dropout_final
        period = float(self.morpho_dropout_end_epoch)
        morpho_dropout = initial + (min(epoch, period) * (final - initial) / period)
        return morpho_dropout

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        src_data_files = glob(split_path + "_{}_".format(SRC_SIDE) + "*")
        tgt_data_files = glob(split_path + "_{}_".format(TGT_SIDE) + "*")
        data_files = src_data_files + tgt_data_files
        if not data_files:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        self.datasets[split] = build_combined_dataset(
            self.src_dict,
            src_data_files,
            self.morpho_dropout_for(epoch) if split == 'train' else 0.0,
            self.tgt_dict,
            tgt_data_files,
            self.args.seed,
            epoch,
        )

    def build_generator(self, models, args):
        # copied from fairseq_task.py to choose our implementation

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        if (
                sum(
                    int(cond)
                    for cond in [
                        sampling,
                        diverse_beam_groups > 0,
                        match_source_len,
                        diversity_rate > 0,
                    ]
                )
                > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        return SequenceGenerator(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
        )
