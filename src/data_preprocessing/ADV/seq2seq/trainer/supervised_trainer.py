from __future__ import division
import logging
import os
import random
import time
import json

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint


class SupervisedTrainer(object):
    """The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
            expt_dir (optional, str): experiment Directory to store details of the experiment,
                    by default it makes a folder in the current directory to store the details (default: `experiment`).
            loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
            batch_size (int, optional): batch size for experiment, (default: 64)
            checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """

    def __init__(
        self,
        expt_dir="experiment",
        loss=NLLLoss(),
        batch_size=64,
        random_seed=None,
        checkpoint_every=1000,
        print_every=100,
        tensorboard=True,
        task_type="generate",
    ):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)
        self.task_type = task_type

    def _train_batch(
        self,
        input_variable,
        input_lengths,
        target_variable,
        model,
        teacher_forcing_ratio,
    ):
        loss = self.loss

        if self.task_type == "classify":
            logits = model(
                input_variable,
                input_lengths,
                target_variable,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(logits, target_variable)
            model.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()

        else:
            decoder_outputs, decoder_hidden, other = model(
                input_variable,
                input_lengths,
                target_variable,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            loss.reset()
            for step, step_output in enumerate(decoder_outputs):
                batch_size = target_variable.size(0)
                loss.eval_batch(
                    step_output.contiguous().view(batch_size, -1),
                    target_variable[:, step + 1],
                )

            model.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.get_loss()

    def _train_epoches(
        self,
        data,
        model,
        n_epochs,
        start_epoch,
        start_step,
        dev_data=None,
        teacher_forcing_ratio=0,
    ):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        device = self.device
        # device = None if torch.cuda.is_available() else -1
        from seq2seq.dataset.dataloader import create_bucket_iterator

        batch_iterator = create_bucket_iterator(
            dataset=data,
            batch_size=self.batch_size,
            sort_key=lambda x: len(x.src),
            device=device,
            train=True,
        )

        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs

        self.print_every = steps_per_epoch // 2

        log.info("Steps per epoch: %d" % steps_per_epoch)
        log.info("Total steps: %d" % total_steps)

        step = start_step
        step_elapsed = 0

        best_f1 = 0.0
        best_acc = 0.0

        Checkpoint(
            model=model,
            optimizer=self.optimizer,
            epoch=0,
            step=step,
            input_vocab=data.fields[seq2seq.src_field_name].vocab,
            output_vocab=data.fields[seq2seq.tgt_field_name].vocab,
        ).save(self.expt_dir, name="Best_F1")

        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            batch_generator = batch_iterator.__iter__()
            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator)

            model.train(True)
            pbar = tqdm(
                batch_generator,
                total=len(batch_iterator),
                desc="Epoch %d" % epoch,
                ncols=100,
            )
            for batch in pbar:
                step += 1
                step_elapsed += 1

                input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
                target_variables = getattr(batch, seq2seq.tgt_field_name)

                loss = self._train_batch(
                        input_variables,
                        input_lengths.tolist(),
                    target_variables,
                    model,
                    teacher_forcing_ratio,
                )

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if True:
                    print_loss_avg = print_loss_total / (step + 1)

                    pbar.set_description(
                        f"Epoch {epoch}, Loss {round(print_loss_avg, 4)}"
                    )

            if step_elapsed == 0:
                continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Train %s: %.4f" % (
                epoch,
                self.loss.name,
                epoch_loss_avg,
            )

            other_metrics = {}
            if dev_data is not None:
                if self.task_type == "classify":
                    d = self.evaluator.evaluate_classify(
                        model, dev_data, device=self.device
                    )

                    with open(os.path.join(self.expt_dir, "test.json"), "w") as f:
                        f.write(json.dumps(d, indent=4))

                    dev_loss = d["metrics"]["Loss"]
                    other_metrics = d["metrics"]

                    self.optimizer.update(dev_loss, epoch)

                else:
                    d = self.evaluator.evaluate_generate(
                        model, dev_data, device=self.device
                    )

                    with open(os.path.join(self.expt_dir, "test.json"), "w") as f:
                        f.write(json.dumps(d, indent=4))

                    dev_loss = d["metrics"]["Loss"]
                    accuracy = d["metrics"]["accuracy (torch)"]
                    other_metrics = d["metrics"]
                    self.optimizer.update(dev_loss, epoch)

                if other_metrics["f1"] > best_f1:
                    Checkpoint(
                        model=model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        step=step,
                        input_vocab=data.fields[seq2seq.src_field_name].vocab,
                        output_vocab=data.fields[seq2seq.tgt_field_name].vocab,
                    ).save(self.expt_dir, name="Best_F1")

                model.train(mode=True)

            else:
                self.optimizer.update(epoch_loss_avg, epoch)

    def train(
        self,
        model,
        data,
        num_epochs=5,
        resume=False,
        dev_data=None,
        optimizer=None,
        teacher_forcing_ratio=0,
        load_checkpoint=None,
        device=None,
    ):
        """Run training for a given model.

        Args:
                model (seq2seq.models): model to run training on, if `resume=True`, it would be
                   overwritten by the model loaded from the latest checkpoint.
                data (seq2seq.dataset.dataset.Dataset): dataset object to train on
                num_epochs (int, optional): number of epochs to run (default 5)
                resume(bool, optional): resume training with the latest checkpoint, (default False)
                dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
                optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
                   (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
                teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
                model (seq2seq.models): trained model.
        """
        assert device
        print(f"device: {device}")
        self.device = device
        # If training is set to resume
        if resume:
            if load_checkpoint is None:
                load_checkpoint = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(load_checkpoint)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop("params", None)
            defaults.pop("initial_lr", None)
            self.optimizer.optimizer = resume_optim.__class__(
                model.parameters(), **defaults
            )

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step

            self.logger.info(
                "Resuming training from %d epoch, %d step" % (start_epoch, step)
            )
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
            self.optimizer = optimizer

        self.logger.info(
            "Optimizer: %s, Scheduler: %s"
            % (self.optimizer.optimizer, self.optimizer.scheduler)
        )

        self._train_epoches(
            data,
            model,
            num_epochs,
            start_epoch,
            step,
            dev_data=dev_data,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return model
