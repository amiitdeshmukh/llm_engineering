# Training Metrics Explained

This lesson explains the training and evaluation terms shown in the terminal
and Weights & Biases while fine-tuning the price model.

## 1. What happens during training?

One row in the price dataset contains a prompt and a correct completion:

```text
Prompt:
What does this product cost?

Steel door knob with mounting hardware.

Price is $

Completion:
25.00
```

The tokenizer converts this text into token IDs. At every completion position,
Qwen produces a score for every possible next token.

Training then performs these operations:

1. Qwen predicts the next completion tokens.
2. The predictions are compared with the correct tokens.
3. A loss value is calculated.
4. Gradients are calculated for the trainable LoRA parameters.
5. The optimizer updates the LoRA parameters.

The original Qwen parameters remain frozen during LoRA training.

## 2. Training loss

Training loss measures how much probability the model assigned to the correct
next tokens in the training completions.

For one correct token, the calculation is approximately:

```text
loss = -log(probability assigned to the correct token)
```

Examples:

```text
Correct-token probability 0.90 -> loss 0.105
Correct-token probability 0.50 -> loss 0.693
Correct-token probability 0.10 -> loss 2.303
Correct-token probability 0.01 -> loss 4.605
```

Therefore:

```text
Lower loss  -> more probability was assigned to correct tokens
Higher loss -> less probability was assigned to correct tokens
```

The first training run logged:

```text
4.227 -> 3.015 -> 2.159
```

This indicates that the model improved at predicting training completion
tokens during the run.

The final `train_loss` was `3.06`. That was the average loss over the complete
training run. The individual logged values represented smaller intervals of
the run.

Training loss does not directly measure price error. A model can improve its
token probabilities while still generating a price that is financially far
from the correct price.

## 3. Mean token accuracy

`mean_token_accuracy` measures how often the model's highest-scoring token was
the correct next token.

Suppose a completion contains four tokens:

```text
Correct:   ["25", ".", "00", EOS]
Predicted: ["20", ".", "00", EOS]
```

Three of the four tokens were correct:

```text
token accuracy = 3 / 4 = 0.75 = 75%
```

The run logged:

```text
0.4757 -> 47.57%
0.5531 -> 55.31%
0.5548 -> 55.48%
```

Validation token accuracy was:

```text
0.5698 -> 56.98%
```

This does not mean that 56.98% of complete prices were correct. One price can
contain multiple tokens. The model may correctly predict punctuation and
common number pieces while generating the wrong complete price.

## 4. Number of tokens

`num_tokens` is the cumulative number of non-padding input tokens processed so
far. It includes prompt and completion tokens.

The run showed:

```text
Step 5:   34,570 tokens
Step 10:  69,160 tokens
Step 15: 103,100 tokens
```

The value always increases because it is a running total.

At global step 5, the training process had handled approximately:

```text
5 updates
* 4 accumulated batches per update
* 16 examples per batch
= 320 examples
```

Those examples contained 34,570 non-padding tokens:

```text
34,570 / 320 = approximately 108 tokens per example
```

The maximum sequence length was 128, but many examples contained fewer than
128 real tokens.

## 5. Learning rate

The learning rate controls the scale of parameter updates.

A simplified update is:

```text
new parameter = old parameter - learning rate * gradient
```

The configured maximum was:

```python
LEARNING_RATE = 1e-4
```

This is equal to:

```text
0.0001
```

The run logged:

```text
Step 5:  0.00009045
Step 10: 0.00004477
Step 15: 0.000004323
```

The learning rate decreased because the run used a cosine scheduler:

```text
warm up to the maximum
-> gradually decrease
-> approach zero near the end
```

A learning rate that is too large can produce unstable updates or remove
useful pretrained behaviour. A learning rate that is too small can make
learning extremely slow.

The learning rate is not the exact amount added to every parameter. AdamW also
uses gradient history, weight decay, and other calculations.

## 6. Gradients

A gradient is calculated for every trainable LoRA parameter.

It describes how the loss would change if that parameter changed. Its sign
contributes to the update direction, and its magnitude describes sensitivity
to that parameter.

The adapter has millions of trainable parameters, so one training update
produces millions of gradient values.

## 7. Gradient norm

`grad_norm` summarizes the combined magnitude of all those gradients in one
number.

Conceptually:

```text
gradient norm = square root of the sum of every squared gradient
```

The run logged:

```text
2.328
3.859
5.125
```

Gradient norm is useful for detecting numerical instability:

```text
Finite changing values     -> calculations are continuing
Large repeated spikes      -> possible instability
NaN or infinity            -> numerical training failure
```

There is no universal ideal gradient norm. It depends on the model, task,
data, optimizer, and other settings.

The configuration used:

```python
max_grad_norm = 0.3
```

This enables gradient clipping. If the calculated gradients are too large,
the trainer scales them down before updating LoRA. The logged value can show
the norm measured before clipping, so the chart can contain values greater
than `0.3`.

Gradient norm is not accuracy or loss. It describes the magnitude of the
current update signal.

## 8. Physical batch size

A physical batch is the group of examples processed by Qwen simultaneously.

The run used:

```python
BATCH_SIZE = 16
```

Therefore, one physical batch contained 16 prompt-completion examples.

After tokenization and padding, tensors could have shapes similar to:

```text
input_ids:      (16, 128)
attention_mask: (16, 128)
labels:         (16, 128)
```

The first number is the batch size. The second is the maximum token sequence
length.

A larger physical batch requires more memory because PyTorch must retain
intermediate calculations for more examples simultaneously.

## 9. Gradient accumulation

The run used:

```python
GRADIENT_ACCUMULATION_STEPS = 4
```

This means:

```text
Batch 1 -> calculate and keep gradients
Batch 2 -> add gradients
Batch 3 -> add gradients
Batch 4 -> add gradients
Then update LoRA once
```

The effective batch size was:

```text
physical batch size * accumulation steps
16 * 4 = 64 examples per optimizer update
```

Gradient accumulation allows several physical batches to contribute to one
update without keeping all their forward calculations in memory at the same
time.

## 10. Global step

`global_step` counts optimizer updates. It does not count individual examples,
tokens, or physical batches.

With the training configuration:

```text
4 physical batches -> 1 optimizer update -> global step increases by 1
```

For 1,000 examples and an effective batch size of 64:

```text
1,000 / 64 = approximately 16 global steps
```

This is why the progress bar showed:

```text
0/16 -> 16/16
```

Most W&B training charts use global step on the horizontal axis.

## 11. Epoch

One epoch means that the model has processed the complete training dataset
once.

The logs contained:

```text
epoch 0.3175 -> approximately 31.75% through the dataset
epoch 0.6349 -> approximately 63.49% through the dataset
epoch 0.9524 -> approximately 95.24% through the dataset
epoch 1.0    -> one complete dataset pass
```

With three epochs, the same training split is processed three times:

```text
Epoch 1 -> all training examples
Epoch 2 -> all training examples again
Epoch 3 -> all training examples again
```

## 12. Entropy

Entropy measures how spread out the model's possible-next-token probabilities
are.

If the model assigns similar probabilities to many possible tokens, entropy
is higher. If it concentrates most probability on fewer tokens, entropy is
lower.

The run logged:

```text
1.727
1.442
1.587
```

Lower entropy can mean greater confidence, but confidence does not guarantee
correctness. A confidently incorrect prediction can have low entropy.

## 13. Training data

Training examples are used to update LoRA parameters.

For a training batch:

1. The model predicts completion tokens.
2. Training loss is calculated.
3. Gradients are calculated.
4. The optimizer changes LoRA parameters.

Metrics starting with `train/` describe this process:

```text
train/loss
train/mean_token_accuracy
train/grad_norm
train/learning_rate
```

## 14. Validation or evaluation data

Validation examples check the current adapter without changing it.

For evaluation:

1. The model predicts validation completion tokens.
2. Evaluation loss and accuracy are calculated.
3. No training gradients are retained.
4. LoRA parameters are not updated.

Metrics starting with `eval/` come from validation examples:

```text
eval/loss
eval/mean_token_accuracy
eval/entropy
```

Validation answers this question:

```text
Did the changes learned from training examples also help on different
examples?
```

## 15. Comparing training and evaluation loss

Common patterns include:

| Pattern | Possible meaning |
|---|---|
| Training loss falls and evaluation loss falls | Learning transfers to unseen examples |
| Training loss falls and evaluation loss rises | Possible overfitting |
| Both losses remain high and flat | Possible underfitting or configuration problem |
| Loss becomes NaN or unstable | Numerical or learning-rate problem |

The first run produced:

```text
Final logged training loss: 2.159
Evaluation loss:            1.844
```

Evaluation loss can be lower because:

- LoRA dropout is active during training but disabled during evaluation.
- Evaluation uses the final trained adapter.
- The selected validation examples may be easier than the training examples.

## 16. Test data

Test data is reserved for the final independent measurement.

| Dataset | Updates LoRA? | Purpose |
|---|---:|---|
| Training | Yes | Learn LoRA parameters |
| Validation | No | Choose settings and monitor generalization |
| Test | No | Measure the final selected model |

Repeatedly choosing training settings based on test results makes the test set
part of the decision process. Hyperparameters should be chosen using validation
results, then the final configuration should be measured on the test set.

## 17. Price error

For this project, the most important task metric is price error:

```text
absolute error = absolute value of (predicted price - actual price)
```

Example:

```text
Predicted price: $80
Actual price:    $65
Absolute error:  $15
```

Useful price metrics include:

```text
Mean absolute error
Median absolute error
Percentage of predictions within $10
Percentage of predictions within $20
```

Training loss and token accuracy measure token prediction. Price error measures
whether the generated number is useful for the actual price task.

## 18. Training speed metrics

Useful speed metrics include:

```text
train_runtime
train_samples_per_second
train_steps_per_second
num_tokens
```

The first run reported approximately:

```text
Training runtime:          1,553 seconds
Samples per second:        0.644
Optimizer steps per second: 0.01
```

These metrics help compare the practical cost of different batch sizes,
sequence lengths, models, and LoRA settings.

## 19. Hardware metrics on Apple Silicon

The Mac CPU and GPU use the same unified memory pool. During training, W&B can
record:

```text
GPU utilization
GPU frequency
GPU power
GPU temperature
System power
Memory usage
```

High GPU utilization means the neural-network calculations are using the MPS
GPU. If temperature rises and GPU frequency later falls, the Mac may be
reducing speed to control temperature.

## 20. What to monitor first

For the price model, prioritize:

1. Training loss.
2. Evaluation loss.
3. Validation mean absolute price error.
4. Validation median absolute price error.
5. Evaluation mean token accuracy.
6. Learning rate.
7. Gradient norm.
8. GPU utilization, temperature, and memory pressure.

The most important model-selection question is:

```text
Which checkpoint has the lowest validation price error?
```

After selecting a checkpoint using validation results, measure it once on the
test dataset to report final performance.
