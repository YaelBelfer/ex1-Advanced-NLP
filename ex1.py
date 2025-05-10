import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from tqdm import tqdm
import wandb
import evaluate
import torch

def encode(examples, truncation):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=truncation, padding='max_length')

def train(args, model, tokenizer):
    wandb.init(project="mrpc-paraphrase-detection", config={
        "epochs": args.num_train_epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "model": "bert-base-uncased"
    })

    # Load and preprocess training set
    train_dataset = load_dataset('glue', 'mrpc', split='train')
    if args.max_train_samples != -1:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    train_dataset = train_dataset.map(lambda x: encode(x, truncation=True), batched=True)
    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Load and preprocess validation set
    val_dataset = load_dataset('glue', 'mrpc', split='validation')
    if args.max_eval_samples != -1:
        val_dataset = val_dataset.select(range(args.max_eval_samples))
    val_dataset = val_dataset.map(lambda x: encode(x, truncation=True), batched=True)
    val_dataset = val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    accuracy_metric = evaluate.load("accuracy")

    global_step = 0
    for epoch in range(args.num_train_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_train_epochs}")
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            wandb.log({"train_loss": loss.item(), "step": global_step})
            global_step += 1

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        results = accuracy_metric.compute(predictions=all_preds, references=all_labels)
        print(f"Validation Accuracy after epoch {epoch+1}: {results['accuracy']:.4f}")
        wandb.log({"val_accuracy": results["accuracy"]}, step=global_step)

    # Save model checkpoint
    base_model_path = f"lr{args.lr}_bs{args.batch_size}_ep{args.num_train_epochs}"
    model.save_pretrained(base_model_path)
    tokenizer.save_pretrained(base_model_path)

    # Write result to res.txt
    res_line = f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {results['accuracy']:.4f}\n"
    with open("res.txt", "a") as f:
        f.write(res_line)


def predict(args, model, tokenizer, output_file="predictions"):
    test_dataset = load_dataset('glue', 'mrpc', split='test')
    if args.max_predict_samples != -1:
        test_dataset = test_dataset.select(range(args.max_predict_samples))

    full_inputs = list(zip(test_dataset['sentence1'], test_dataset['sentence2']))

    test_dataset = test_dataset.map(lambda x: encode(x, truncation=False), batched=True)
    test_dataset = test_dataset.map(lambda x: {'labels': x['label']}, batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval().to(device)

    predictions = []
    gold_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            gold_labels.extend(batch["labels"].cpu().numpy())

    with open(f'{output_file}.txt', "w") as f:
        for (s1, s2), label in zip(full_inputs, predictions):
            f.write(f"{s1}###{s2}###{label}\n")

    print(f"Predictions saved to {output_file}")

    accuracy_metric = evaluate.load("accuracy")
    results = accuracy_metric.compute(predictions=predictions, references=gold_labels)
    print(f"Test Accuracy: {results['accuracy']:.4f} for {args.model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Command line interface for training, evaluation, and prediction")

    parser.add_argument("--max_train_samples", type=int, default=-1,
                        help="Number of training samples to use or -1 to use all")
    parser.add_argument("--max_eval_samples", type=int, default=-1,
                        help="Number of validation samples to use or -1 to use all")
    parser.add_argument("--max_predict_samples", type=int, default=-1,
                        help="Number of prediction samples to use or -1 to use all")

    parser.add_argument("--num_train_epochs", type=int, required=True,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, required=True,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, required=True,
                        help="Training batch size")

    parser.add_argument("--do_train", action="store_true",
                        help="Flag to run training")
    parser.add_argument("--do_predict", action="store_true",
                        help="Flag to run prediction and generate predictions.txt")

    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the model for prediction")

    return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  model_name = "bert-base-uncased"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

  if args.do_train:
    train(args, model, tokenizer)
  elif args.do_predict:
    predict(args, model, tokenizer)