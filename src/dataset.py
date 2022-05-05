from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import os 
from itertools import cycle
from utils.args import get_argument_parser

class MSMARCO_Triple(IterableDataset):
    def __init__(self, args, data_path) -> None:
        """
        iterably load the triples, tokenize and return
        """
        self.args = args
        super().__init__()

        self.query_length = args.query_length
        self.sequence_length = args.sequence_length

        self.triple_path = os.path.join(args.data_root, data_path)
        
        self.length = sum(1 for line in open(self.triple_path))
        self.tokenizer = AutoTokenizer.from_pretrained(args.plm_dir)

    def __len__(self):
        return self.length

    def _parse_triple_file(self):
        with open(self.triple_path, "r") as f:
            for line in f:
                query, pos_seq, neg_seq = line.strip().split("\t")

                query_output = self.tokenizer(
                    query, 
                    return_tensors="pt", 
                    padding="max_length", 
                    max_length=self.query_length, 
                    truncation=True
                    )

                pos_seq_output = self.tokenizer(
                    pos_seq, 
                    return_tensors="pt", 
                    padding="max_length", 
                    max_length=self.sequence_length, 
                    truncation=True
                    )

                neg_seq_output = self.tokenizer(
                    neg_seq, 
                    return_tensors="pt", 
                    padding="max_length", 
                    max_length=self.sequence_length, 
                    truncation=True
                    )

                return_dict = {
                    "q_feats": query_output,
                    "pos_feats": pos_seq_output,
                    "neg_feats": neg_seq_output,
                }
                yield return_dict

    def __iter__(self):
        return cycle(self._parse_triple_file())

class Text_Dataset(IterableDataset):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.plm_dir)
        self.max_length = args.sequence_length
        self.data_path = os.path.join(args.data_root, args.corpus_path)

    def __iter__(self):
        for line in open(self.data_path):
            feat = self.tokenizer(line, 
                                    return_tensors="pt", 
                                    padding="max_length", 
                                    max_length=self.max_length, 
                                    truncation=True)

            yield feat


if __name__ == "__main__":
    args = get_argument_parser()
    DL = MSMARCO_Triple(args)
    # from transformers import AutoModel
    # encoder = AutoModel.from_pretrained(args.plm_dir, return_dict=True)
    # for i,sample in enumerate(DL):
    #     print(sample["q_feats"]["input_ids"].shape)
    #     if i > 2: break
    print(len(DL))