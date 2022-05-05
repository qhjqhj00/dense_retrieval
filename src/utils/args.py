import argparse
import os

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-hnn", "--hard-negative-num", help="number of hard negatives", type=int, default=1)
    parser.add_argument("-ehn", "--enable-hard-negative", help="use hard negatives", action="store_true", default=False)
    parser.add_argument("-ein", "--enable-inbatch-negative", help="use in-batch negatives", action="store_true", default=False)
    parser.add_argument("-e", "--epoch",help="epochs to train the model", type=int, default=40)
    parser.add_argument("-d", "--device", help="gpu index, -1 for cpu", type=lambda x: int(x) if x != "cpu" else "cpu", default=0)
    parser.add_argument("-bs", "--batch-size", help="batch size in training", type=int, default=32)
    parser.add_argument("-bse", "--batch-size-encode", help="batch size in encoding", type=int, default=512)
    parser.add_argument("-ck","--checkpoint", help="load the model from checkpoint before training/evaluating", type=str, default="none")
    parser.add_argument("-vs","--validate-step", help="evaluate and save the model every step", type=str, default="0")
    parser.add_argument("-hs","--hold-step", help="don't evaluate until reaching hold step", type=str, default="0")
    parser.add_argument("-sav","--save-at-validate", help="save the model every time of validating", action="store_true", default=False)
    parser.add_argument("-vb","--verbose", help="variant's name", type=str, default=None)
    parser.add_argument("--dropout", help="dropout probability", type=float, default=0.1)
    parser.add_argument("-lr", "--learning-rate", help="learning rate", type=float, default=3e-6)
    parser.add_argument("-sch", "--scheduler", help="choose schedule scheme for optimizer", choices=["linear","none"], default="none")
    parser.add_argument("--warmup", help="warmup steps of scheduler", type=float, default=0.1)
    parser.add_argument("-cln", "--clip-grad-norm", help="max norm of clipping gradients", type=float, default=0.)
    parser.add_argument("-it", "--index-type", help="index type", choices=["anserini", "flat", "invvec", "opq", "ivfflat", "ivfpq", "ivfopq", "hnsw", "hnswopq", "distillflat", "distillopq", "uni", "hybrid", "invhit"], default="flat")
    parser.add_argument("-hits", help="hit number per query", type=int, default=1000)
    parser.add_argument("-bc", "--build-collection", action="store_true", default=False)
    parser.add_argument("-op","--output-path", help="", type=str, default="./output/")
    parser.add_argument("-wd", "--weight-decay", help="weight decay of AdamW", type=float, default=0.01)
    parser.add_argument("-dr", "--data-root", type=str, default="../data")

    parser.add_argument("-tf", "--train-file", type=str, default="MSMARCO-Passage/train.tsv")
    parser.add_argument("-df", "--dev-file", type=str, default="MSMARCO-Passage/dev.tsv")
    parser.add_argument("-cp", "--corpus-path", help="path to corpus", type=str, default="corpus.txt")

    parser.add_argument("-ql", "--query-length", help="query token length", type=int, default=32)
    parser.add_argument("-sl", "--sequence-length", help="sequence token length", type=int, default=128)
    parser.add_argument("-plm", help="short name of pre-trained language models", type=str, default="bert")

    args = parser.parse_args()

    if  args.enable_hard_negative:
        args.enable_inbatch_negative = False
        args.hard_negative_num = 1

    args.plm_dir = os.path.join(args.data_root, "PLM", args.plm)

    return args

if __name__ == "__main__":
    args = get_argument_parser()
    print(args)