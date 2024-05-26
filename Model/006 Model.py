import logging
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, BertConfig
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
import argparse
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, label=None):
        self.guid = guid
        self.text_a = text_a
        self.label = label


def read_examples(file_path, is_training):
    df = pd.read_csv(file_path)
    examples = []

    if is_training and 'id' in df.columns and 'label' in df.columns:
        for val in df[['id', 'TextBody', 'label']].values:
            guid = val[0]
            text_a = val[1]
            label = val[2]
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
    else:
        for val in df[['Date', 'Year', 'TextBody','id']].values:
            guid = val[3]
            text_a = val[0]
            examples.append(InputExample(guid=guid, text_a=text_a, label=None))

    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training):
    features = []
    for example in examples:
        tokens = tokenizer.tokenize(example.text_a)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:max_seq_length - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        # Pad the input tokens
        padding_length = max_seq_length - len(input_ids)
        input_ids += [0] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [0] * padding_length

        if is_training:
            label = int(example.label)
        else:
            label = -1

        features.append((input_ids, input_mask, segment_ids, label))
    return features

def load_and_cache_examples(file_path, tokenizer, max_seq_length, is_training):
    examples = read_examples(file_path, is_training)
    features = convert_examples_to_features(examples, tokenizer, max_seq_length, is_training)
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f[3] for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def plot_metrics(train_metrics, val_metrics, metric_name, plot_dir):
    epochs = range(1, len(train_metrics) + 1)
    plt.figure()
    plt.plot(epochs, train_metrics, 'bo-', label=f'Training {metric_name}')
    plt.plot(epochs, val_metrics, 'ro-', label=f'Validation {metric_name}')
    plt.title(f'{metric_name.capitalize()} over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.savefig(f'{plot_dir}/{metric_name}_curve.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--plot_dir", default=None, type=str, required=True)
    parser.add_argument("--pred_file", default=None, type=str, required=True)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--train_file", default="train.csv", type=str)
    parser.add_argument("--eval_file", default="dev.csv", type=str)
    parser.add_argument("--test_file", default="test.csv", type=str)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=5.0, type=float)
    parser.add_argument("--save_steps", default=500, type=int)
    parser.add_argument("--seed", type=int, default=42)

    # Training configuration
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation steps.")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-6, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay if we apply some.")
    parser.add_argument("--drop_out_rate", type=float, default=0, help="drop out rate.")
    parser.add_argument("--freeze", type=int, default=0, help="Whether to freeze BERT parameters.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    # Setup CUDA/mps & distributed training
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load pretrained model and tokenizer
    config = BertConfig.from_pretrained(
    "hfl/rbtl3",
    hidden_dropout_prob=args.drop_out_rate,
    attention_probs_dropout_prob=args.drop_out_rate)
    config.num_labels = 3  # Update this based on your dataset

    tokenizer = BertTokenizer.from_pretrained('hfl/rbtl3')
    model = BertForSequenceClassification.from_pretrained('hfl/rbtl3', config=config)
    model.to(device)

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    test_losses = []
    test_accuracies = []
    test_f1_scores = []

    if args.do_train:
        dataset = load_and_cache_examples(os.path.join(args.data_dir, args.train_file), tokenizer, args.max_seq_length, is_training=True)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size)

        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.per_gpu_eval_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        total_steps = len(train_dataloader) * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

        best_val_loss = float('inf')

        for epoch in range(int(args.num_train_epochs)):
            model.train()
            epoch_loss, epoch_accuracy, epoch_f1 = 0, 0, 0
            nb_steps, nb_examples = 0, 0

            for step, batch in enumerate(train_dataloader):
                input_ids, attention_mask, segment_ids, labels = [b.to(device) for b in batch]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # Gradient clipping
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                preds = torch.argmax(logits, dim=-1)
                correct_predictions = (preds == labels).sum().item()
                total_predictions = labels.size(0)

                epoch_loss += loss.item()
                epoch_accuracy += correct_predictions
                epoch_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

                nb_steps += 1
                nb_examples += total_predictions

            avg_loss = epoch_loss / nb_steps
            avg_accuracy = epoch_accuracy / nb_examples
            avg_f1 = epoch_f1 / nb_steps

            train_losses.append(avg_loss)
            train_accuracies.append(avg_accuracy)
            train_f1_scores.append(avg_f1)

            logger.info(f"Epoch {epoch + 1}/{int(args.num_train_epochs)}")
            logger.info(f"Train loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, F1: {avg_f1:.4f}")

            # Validation
            model.eval()
            eval_loss, eval_accuracy, eval_f1 = 0, 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            total_eval_loss = 0
            correct_predictions = 0
            total_predictions = 0
            all_labels = []
            all_preds = []

            for step, batch in enumerate(val_dataloader):
                input_ids, attention_mask, segment_ids, labels = [b.to(device) for b in batch]

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                total_eval_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            avg_eval_loss = total_eval_loss / len(val_dataloader)
            avg_eval_accuracy = correct_predictions / total_predictions
            avg_eval_f1 = f1_score(all_labels, all_preds, average='macro')

            val_losses.append(avg_eval_loss)
            val_accuracies.append(avg_eval_accuracy)
            val_f1_scores.append(avg_eval_f1)

            logger.info(f"Validation Loss: {avg_eval_loss:.4f}, Accuracy: {avg_eval_accuracy:.4f}, F1: {avg_eval_f1:.4f}")

            if avg_eval_loss < best_val_loss:
                best_val_loss = avg_eval_loss
                patience_counter = 0
                best_checkpoint_path = os.path.join(args.output_dir, 'best_model')
                if not os.path.exists(best_checkpoint_path):
                    os.makedirs(best_checkpoint_path)
                model.save_pretrained(best_checkpoint_path)
                tokenizer.save_pretrained(best_checkpoint_path)
                logger.info("Saving best model checkpoint to %s", os.path.join(args.output_dir, 'best_model'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info("Early stopping triggered")
                    break

    if args.do_eval:
        test_dataset = load_and_cache_examples(os.path.join(args.data_dir, args.eval_file), tokenizer,
                                               args.max_seq_length, is_training=True)
        test_labels = test_dataset.tensors[3].cpu().numpy()
        test_dataloader = DataLoader(test_dataset, batch_size=args.per_gpu_eval_batch_size, shuffle=False)

        model = BertForSequenceClassification.from_pretrained(os.path.join(args.output_dir, 'best_model'))
        model.to(device)

        model.eval()
        eval_loss, eval_accuracy, eval_f1 = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        total_test_loss = 0
        correct_predictions = 0
        total_predictions = 0
        all_labels = []
        all_preds = []

        for step, batch in enumerate(test_dataloader):
            input_ids, attention_mask, segment_ids, labels = [b.to(device) for b in batch]

            labels = labels.long()

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

            total_test_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        avg_test_loss = total_test_loss / len(test_dataloader)
        avg_test_accuracy = correct_predictions / total_predictions
        avg_test_f1 = f1_score(all_labels, all_preds, average='macro')

        test_losses.append(avg_test_loss)
        test_accuracies.append(avg_test_accuracy)
        test_f1_scores.append(avg_test_f1)

        logger.info(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {avg_test_accuracy:.4f}, F1: {avg_test_f1:.4f}")

    plot_metrics(train_losses, val_losses, 'loss', args.plot_dir)
    plot_metrics(train_accuracies, val_accuracies, 'accuracy', args.plot_dir)
    plot_metrics(train_f1_scores, val_f1_scores, 'f1', args.plot_dir)


    if args.do_predict:
    # Load the best model
      model = BertForSequenceClassification.from_pretrained(os.path.join(args.output_dir, 'best_model'))
      model.to(device)
      model.eval()

      # Load the tokenizer
      tokenizer = BertTokenizer.from_pretrained(os.path.join(args.output_dir, 'best_model'))

      # Load and preprocess the new dataset for prediction
      new_data_file = args.pred_file
      new_df = pd.read_csv(os.path.join(args.data_dir, new_data_file))

      predictions_by_year = {}

      for year, group in new_df.groupby('Year'):
          # Preprocess the text data
          examples = [InputExample(guid=None, text_a=text, label=-1) for text in group['TextBody'].values]
          features = convert_examples_to_features(examples, tokenizer, args.max_seq_length, is_training=False)

          all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
          all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
          all_segment_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
          all_label_ids = torch.tensor([f[3] for f in features], dtype=torch.long)

          dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
          dataloader = DataLoader(dataset, batch_size=args.per_gpu_eval_batch_size, shuffle=False)

          predictions = []
          for batch in dataloader:
              input_ids, attention_mask, segment_ids, _ = [b.to(device) for b in batch]

              with torch.no_grad():
                  outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                  logits = outputs.logits

              preds = torch.argmax(logits, dim=-1)
              predictions.extend(preds.cpu().numpy())

          predictions_by_year[year] = predictions

      # Save predictions to a file
      output_prediction_file = os.path.join(args.output_dir, "predictions_by_year.txt")
      with open(output_prediction_file, "w") as writer:
          for year, preds in predictions_by_year.items():
              writer.write(f"Year: {year}\n")
              for pred in preds:
                  writer.write(f"{pred}\n")
              writer.write("\n")

      logger.info(f"Predictions saved to {output_prediction_file}")


if __name__ == "__main__":
    main()