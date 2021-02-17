import argparse
from morphodropout.task import MorphoTranslation
from morphodropout.model import MorphoTransformer


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    MorphoTranslation.add_args(parser)
    MorphoTransformer.add_args(parser)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    task = MorphoTranslation.setup_task(args)
    task.load_dataset('train')
    dataset = task.dataset('train')

    batch = dataset.collater([dataset[132], dataset[241]])
    # print("--->{}".format(batch['net_input']['src_tokens']))
    model = MorphoTransformer.build_model(args, task)

    result = model(**batch['net_input'])
    print("**** {}".format(result))


if __name__ == '__main__':
    main()
