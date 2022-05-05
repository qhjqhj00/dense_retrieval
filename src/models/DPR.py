from numpy import squeeze
import torch
import torch.nn as nn
from transformers import AutoModel
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

class DPR(pl.LightningModule):
    def __init__(self, 
                 args,
                 train_data,
                 dev_data):

        super(DPR, self).__init__()
        self.args = args
        self.train_data = train_data
        self.dev_data = dev_data
        self.q_encoder = AutoModel.from_pretrained(args.plm_dir, return_dict=True)
        self.ctx_encoder = AutoModel.from_pretrained(args.plm_dir, return_dict=True)
        self.loss = nn.CrossEntropyLoss()

    @staticmethod
    def get_representation(sub_model: nn.Module, feat: dict, fix_encoder: bool = False):
        feat = {k:v.squeeze() for k,v in feat.items()}
        if fix_encoder:
            with torch.no_grad():
                pooled_output = sub_model(**feat).pooler_output

            if sub_model.training:
                pooled_output.requires_grad_(requires_grad=True)
        else:
            pooled_output = sub_model(**feat).pooler_output
        return pooled_output

    def forward(self, batch):
        q_vectors = self.get_representation(self.q_encoder, batch["q_feats"])
        pos_vectors = self.get_representation(self.ctx_encoder, batch["pos_feats"])
        pos_scores = torch.einsum("md, nd-> mn", q_vectors, pos_vectors)
        
        bs = q_vectors.size(0)
        labels = torch.arange(bs, device=q_vectors.device)

        if self.args.enable_hard_negative:
            neg_vectors = self.get_representation(self.ctx_encoder, batch["neg_feats"])
            neg_scores = torch.einsum("md, nd-> mn", q_vectors, neg_vectors)
            scores = torch.cat([pos_scores, neg_scores], dim=-1)
        else:
            scores = pos_scores
        
        return scores, labels
    
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.args.batch_size,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_data,
            batch_size=self.args.batch_size,
            num_workers=2
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, self.args.warmup, len(self.train_data)*self.args.epoch
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_nb):
        scores, labels = self(batch)
        loss = self.loss(scores, labels)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_nb):
        scores, labels = self(batch)
        loss = self.loss(scores, labels)
        _, max_idxs = torch.max(scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(labels).to(max_idxs.device)).sum()
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"total": scores.size(0), "correct": correct_predictions_count}

    def validation_epoch_end(self, outputs):
        total_count = 0
        correct_count = 0
        for output in outputs:
            total_count += output["total"]
            correct_count += output["correct"]
        self.log(
            "correct prediction",
            float(correct_count),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "correct prediction %",
            correct_count / total_count,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def predict_step(self, batch, batch_nb, dataloader_idx=0):
        """
        build ANN index with faiss
        """
        ctx_vectors = self.get_representation(self.ctx_encoder, batch)
        return ctx_vectors
