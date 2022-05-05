from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl

from utils import get_argument_parser, MSMARCO_Triple
from models import DPR

def main():
    args = get_argument_parser()
    
    learning_rate_callback = LearningRateMonitor()
    
    train_data = MSMARCO_Triple(args, args.train_file)
    val_data = MSMARCO_Triple(args, args.dev_file)
    net = DPR(args, train_data, val_data)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_path,
        verbose=True,
        every_n_epochs=1,
        monitor="val_loss",
        save_top_k=-1,
        mode="min",
        save_last=True
    )

    trainer = pl.Trainer(
        accelerator="auto",
        default_root_dir=args.output_path,
        gradient_clip_val=5,
        max_epochs=args.epoch,
        strategy="ddp",
        auto_scale_batch_size=True,
        check_val_every_n_epoch=1,
        callbacks=[learning_rate_callback, checkpoint_callback],
        precision=16
    )

    trainer.fit(net)

if __name__ == "__main__":
    main()
