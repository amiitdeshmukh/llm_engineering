# How LLM Training Works, Step by Step

This lesson follows one product example through Qwen during QLoRA training.
It explains the complete path from text to a LoRA parameter update, without
assuming prior knowledge of neural-network training.

The model and settings discussed here are based on the experiment in this
project:

```text
Base model: Qwen/Qwen2.5-3B-Instruct
Transformer layers: 36
Hidden size: 2,048
Vocabulary size: 151,936
Query attention heads: 16
Key/value heads: 2
Head size: 128
MLP intermediate size: 11,008
LoRA rank: 32
LoRA alpha: 64
LoRA targets: q_proj, k_proj, v_proj, o_proj
Maximum sequence length: 128 tokens
Physical batch size: 16 examples
Gradient accumulation: 4 physical batches
```

## 1. Start with one dataset row

One training row contains a prompt and a correct completion:

```text
Prompt:
What does this cost to the nearest dollar?

Steel door knob with mounting hardware.

Price is $

Completion:
25.00
```

The prompt is the information given to the model. The completion is the text
the model should learn to generate after the prompt.

During prediction, the completion is not provided. During training, it is
provided so the program can calculate how wrong the model's predictions are.

## 2. Select a physical batch

The physical batch size is 16:

```python
BATCH_SIZE = 16
```

The data loader selects 16 prompt-completion rows. These 16 examples will be
processed by Qwen at the same time.

A physical batch is not an optimizer update in this experiment. Four physical
batches are accumulated before one optimizer update. That part is explained
later.

## 3. Tokenize the prompt and completion

Qwen cannot directly calculate with Python strings. Its tokenizer converts
text pieces into token IDs.

A small illustrative result could look like:

```text
Text:      "Price is $25.00"
Token IDs: [10234, 374, 400, 17, 13, 410]
```

The real token IDs depend on Qwen's vocabulary.

For training, the prompt and completion form one sequence:

```text
What does this cost ... Price is $25.00
```

An end-of-sequence token is added after the completion. This tells the model
where the answer ends.

## 4. Build the labels

Labels specify the correct next tokens used to calculate loss.

Consider this simplified token sequence:

```text
Position 0: Product
Position 1: Steel
Position 2: knob
Position 3: Price
Position 4: $
Position 5: 25
Position 6: .00
Position 7: EOS
```

The prompt provides context. The completion provides the answer that should be
scored. Prompt positions are therefore ignored by the loss calculation:

```text
Input tokens:
[Product, Steel, knob, Price, $, 25, .00, EOS]

Labels:
[-100, -100, -100, -100, -100, 25, .00, EOS]
```

`-100` is not sent into Qwen as a token. It is a special value understood by
the loss function:

```text
-100 means: do not calculate loss for this label position
```

The prompt is still read by Qwen and affects its predictions. It is ignored
only when the loss is calculated.

## 5. Understand next-token labels

A causal language model predicts the next token using the tokens before it.

After reading:

```text
Product Steel knob Price $
```

the correct next token is:

```text
25
```

After reading:

```text
Product Steel knob Price $ 25
```

the correct next token is:

```text
.00
```

After reading the complete price, the correct next token is `EOS`.

Transformers performs the required one-position shift internally when it
compares model predictions with labels.

## 6. Truncate long examples

The configured maximum sequence length is 128:

```python
MAX_SEQUENCE_LENGTH = 128
```

If a prompt and completion contain more than 128 tokens, the sequence is
shortened.

If truncation removes every completion token, all labels become `-100`. Such
an example contains no answer token on which to calculate loss, so TRL removes
it from training or evaluation.

## 7. Pad examples in the batch

The 16 examples can have different lengths:

```text
Example A: 108 tokens
Example B:  95 tokens
Example C: 121 tokens
```

One tensor must have the same number of positions in every row. The tokenizer
adds padding to the shorter examples:

```text
Example A: 108 real tokens + 13 padding positions
Example B:  95 real tokens + 26 padding positions
Example C: 121 real tokens
```

The batch could now have this shape:

```text
16 examples x 121 token positions
```

The attention mask identifies real and padded positions:

```text
1 -> real token
0 -> padding
```

Example:

```text
Input IDs:      [21, 32, 43, PAD, PAD]
Attention mask: [ 1,  1,  1,   0,   0]
```

Padding labels are also set to `-100` so they produce no loss.

At this point, the batch contains:

```text
input_ids      -> token numbers Qwen will read
attention_mask -> real-token and padding positions
labels         -> correct completion tokens used for loss
```

## 8. Convert token IDs into embeddings

A token ID is only an integer. Qwen has an embedding table that stores a row
of learned numbers for each token ID.

A tiny example with three numbers per token could be:

```text
Token ID 21 -> [ 0.2, -0.4,  0.7]
Token ID 32 -> [-0.1,  0.5,  0.3]
Token ID 43 -> [ 0.8,  0.1, -0.2]
```

Qwen2.5-3B uses 2,048 numbers for each token representation:

```text
one token ID -> embedding lookup -> 2,048 numbers
```

For a full-sized batch, the embedding output can have a shape such as:

```text
16 examples x 128 positions x 2,048 numbers
```

The embedding values were learned during Qwen's original pretraining. They
remain frozen during this LoRA experiment.

## 9. Enter Transformer layer 1

Qwen contains 36 Transformer layers. The output of one layer becomes the
input to the next layer.

Each layer has two main calculation sections:

1. Self-attention
2. MLP, also called the feed-forward section

Normalization and residual additions are placed around these sections.

We will follow one token representation through the first layer.

## 10. Normalize the input numbers

Before attention, Qwen applies RMS normalization to each token's 2,048
numbers.

Normalization adjusts their overall scale so that later calculations remain
numerically stable. It does not change the number of values:

```text
2,048 input numbers -> RMSNorm -> 2,048 normalized numbers
```

The normalized numbers enter the Query, Key, and Value projections.

## 11. Create Query numbers

`q_proj` converts each token's 2,048 numbers into Query numbers:

```text
2,048 inputs -> q_proj -> 2,048 Query numbers
```

Your LoRA adapter targets `q_proj`, so the calculation is:

```text
Query = original_q_proj(x) + scaled_q_LoRA_adjustment(x)
```

Using matrix names:

```text
Query = Wq(x) + (alpha / rank) * Bq(Aq(x))
```

`Wq` is the frozen original Qwen matrix. `Aq` and `Bq` are trained LoRA
matrices.

## 12. Create Key numbers

`k_proj` converts each token's 2,048 numbers into 256 Key numbers:

```text
2,048 inputs -> k_proj -> 256 Key numbers
```

LoRA is also active here:

```text
Key = Wk(x) + (alpha / rank) * Bk(Ak(x))
```

Key numbers participate in comparisons with Query numbers.

## 13. Create Value numbers

`v_proj` converts each token's 2,048 numbers into 256 Value numbers:

```text
2,048 inputs -> v_proj -> 256 Value numbers
```

LoRA is active here as well:

```text
Value = Wv(x) + (alpha / rank) * Bv(Av(x))
```

Query and Key numbers determine attention weights. Value numbers contain the
information combined using those weights.

## 14. Split Query, Key, and Value into heads

Qwen has:

```text
16 Query heads
2 Key heads
2 Value heads
128 numbers per head
```

The 2,048 Query numbers are divided into 16 groups:

```text
2,048 / 16 = 128 numbers per Query head
```

The 256 Key and Value numbers are each divided into two groups:

```text
256 / 2 = 128 numbers per Key or Value head
```

Several Query heads share each Key/Value head. This design is called grouped
query attention. It reduces the Key/Value memory and computation while keeping
16 Query heads.

For learning the core attention calculation, we can study one Query head and
its associated Key and Value head.

## 15. Apply position information

Attention needs to distinguish token order. Qwen applies rotary position
information to Query and Key numbers.

This changes Query and Key values according to their token positions. The
model can therefore distinguish these sequences:

```text
steel door knob
knob door steel
```

even though both contain the same three words.

Values are not transformed by rotary position information.

## 16. Compare one Query with every allowed Key

Consider four tokens:

```text
Position 0: Steel
Position 1: door
Position 2: knob
Position 3: costs
```

Focus on the Query for `costs`. In a tiny example, let it be:

```text
Query("costs") = [1, 2, 0]
```

Suppose the Keys are:

```text
Key("Steel") = [0, 1, 0]
Key("door")  = [1, 0, 1]
Key("knob")  = [1, 1, 0]
Key("costs") = [0, 0, 1]
```

The Query is compared with each Key using a dot product.

For `Steel`:

```text
1 * 0 + 2 * 1 + 0 * 0 = 2
```

For `door`:

```text
1 * 1 + 2 * 0 + 0 * 1 = 1
```

For `knob`:

```text
1 * 1 + 2 * 1 + 0 * 0 = 3
```

For `costs`:

```text
1 * 0 + 2 * 0 + 0 * 1 = 0
```

The resulting attention scores are:

```text
Steel -> 2
door  -> 1
knob  -> 3
costs -> 0
```

## 17. Scale the attention scores

The real head contains 128 numbers. Qwen divides the dot-product scores by the
square root of the head size:

```text
scaled score = dot product / sqrt(128)
```

Scaling keeps dot products from growing too large as the head dimension grows.
This helps softmax remain numerically usable.

## 18. Apply the causal and padding masks

Qwen is a causal language model. A token may use information from itself and
earlier positions, but not from future positions.

For example, when calculating the representation at position 2:

```text
Position 0 -> allowed
Position 1 -> allowed
Position 2 -> allowed
Position 3 -> blocked because it is in the future
```

Padding positions are also blocked using the attention mask.

Blocked scores are changed so that their softmax weights become zero.

## 19. Convert scores into attention weights

Softmax converts the allowed scores into positive weights that add up to 1.

For the tiny example, the weights could be:

```text
Steel -> 0.24
door  -> 0.09
knob  -> 0.64
costs -> 0.03
```

The total is:

```text
0.24 + 0.09 + 0.64 + 0.03 = 1.00
```

A larger Query-Key score produces a larger attention weight.

## 20. Combine the Value numbers

Suppose `v_proj` produced these tiny Values:

```text
Value("Steel") = [0.2, 0.8]
Value("door")  = [0.5, 0.1]
Value("knob")  = [0.9, 0.4]
Value("costs") = [0.1, 0.3]
```

Multiply each Value by its attention weight:

```text
Steel: 0.24 * [0.2, 0.8] = [0.048, 0.192]
door:  0.09 * [0.5, 0.1] = [0.045, 0.009]
knob:  0.64 * [0.9, 0.4] = [0.576, 0.256]
costs: 0.03 * [0.1, 0.3] = [0.003, 0.009]
```

Add the weighted Values:

```text
First number:
0.048 + 0.045 + 0.576 + 0.003 = 0.672

Second number:
0.192 + 0.009 + 0.256 + 0.009 = 0.466
```

The tiny attention result is:

```text
[0.672, 0.466]
```

The `knob` Value contributed the most because it had the largest weight.

LoRA can change this result in two ways:

```text
Query and Key adjustments -> change the attention weights
Value adjustments         -> change the information being combined
```

## 21. Repeat attention for every head

The previous calculation described one head for one current token position.

Qwen performs the corresponding calculation for:

```text
every allowed token position
every Query head
every example in the batch
```

The 16 Query-head results, each containing 128 numbers, are joined:

```text
16 heads * 128 numbers = 2,048 numbers
```

## 22. Run the output projection

The joined attention result enters `o_proj`:

```text
2,048 attention numbers -> o_proj -> 2,048 output numbers
```

Your adapter targets `o_proj`, so LoRA runs here too:

```text
attention output = Wo(x) + (alpha / rank) * Bo(Ao(x))
```

The original `Wo` matrix remains frozen. The trained `Ao` and `Bo` matrices
add the learned price-task adjustment.

## 23. Add the attention residual

The layer retains the representation that entered attention and adds the
attention output to it:

```text
new representation = old representation + attention output
```

This is called a residual connection. It keeps the previous representation
available while adding the attention result.

The number of values per token remains 2,048.

## 24. Normalize before the MLP

Qwen applies another RMS normalization before the MLP calculations:

```text
2,048 numbers -> RMSNorm -> 2,048 normalized numbers
```

## 25. Run the MLP

Qwen's MLP contains three projections:

```text
gate_proj: 2,048 -> 11,008
up_proj:   2,048 -> 11,008
down_proj: 11,008 -> 2,048
```

The gate path uses an activation function. A simplified expression is:

```text
gate = SiLU(gate_proj(x))
up = up_proj(x)
combined = gate * up
mlp_output = down_proj(combined)
```

The multiplication is performed position by position across the 11,008
numbers.

In the lite training configuration, LoRA targets only the attention
projections. It does not target `gate_proj`, `up_proj`, or `down_proj`.
These original MLP weights remain frozen and receive no LoRA adjustment.

## 26. Add the MLP residual

The MLP output is added to the representation that entered the MLP section:

```text
layer output = representation after attention + MLP output
```

Each token still has 2,048 numbers.

Transformer layer 1 is now complete.

## 27. Repeat through all 36 layers

Layer 1's output becomes layer 2's input. The same general sequence repeats:

```text
normalize
-> q_proj, k_proj, v_proj with LoRA
-> attention scores and Value combination
-> o_proj with LoRA
-> residual addition
-> normalize
-> MLP
-> residual addition
```

This happens through all 36 Transformer layers.

LoRA is active in four attention projections in every layer:

```text
36 layers * 4 targeted projections = 144 LoRA-equipped modules
```

## 28. Apply final normalization

After layer 36, Qwen applies a final RMS normalization to every token's 2,048
numbers.

The shape remains:

```text
batch size x sequence length x 2,048
```

## 29. Calculate vocabulary scores

The language-model output projection converts each token position's 2,048
numbers into one score for every vocabulary entry:

```text
2,048 numbers -> 151,936 vocabulary scores
```

These raw scores are called logits.

For one position, a tiny vocabulary example could be:

```text
Token "20" -> score 2.1
Token "25" -> score 4.8
Token "30" -> score 1.7
Token "40" -> score 0.5
```

Softmax can convert these scores into probabilities. Training needs the
probability assigned to the correct next token.

## 30. Keep only positions with valid labels

The model produced next-token scores at every sequence position. The loss is
calculated only where the label is not `-100`.

Therefore:

```text
Prompt positions     -> model calculates outputs, but loss ignores them
Completion positions -> model calculates outputs and loss scores them
Padding positions    -> model calculates no useful result and loss ignores them
```

The prompt still affects completion predictions through the Transformer
calculations.

## 31. Calculate cross-entropy loss

For each valid completion position, training checks the probability assigned
to the correct token.

The per-token loss is approximately:

```text
loss = -log(probability assigned to the correct token)
```

Examples:

```text
Correct-token probability 0.90 -> loss 0.105
Correct-token probability 0.10 -> loss 2.303
Correct-token probability 0.01 -> loss 4.605
```

Losses from valid completion positions in the batch are combined into one
batch loss value.

Lower loss means the model assigned more probability to the correct
completion tokens.

## 32. Begin backpropagation

The forward pass produced the loss. Backpropagation calculates how that loss
would change if each trainable LoRA parameter changed.

The calculated value for one trainable parameter is its gradient.

Backpropagation follows the calculations in reverse order:

```text
loss
-> vocabulary output calculations
-> layer 36
-> layer 35
-> ...
-> layer 1
-> LoRA parameters in targeted projections
```

## 33. Frozen weights still participate in backpropagation

The original Qwen weights are used in the forward calculations. During
backpropagation, the program passes gradient information through operations
involving those weights so it can reach earlier LoRA modules.

However, the original weights have training disabled:

```text
Original Qwen weights -> no parameter gradients stored, no optimizer updates
LoRA A/B weights      -> gradients stored, optimizer updates allowed
```

The base weights participate in calculation without being modified.

## 34. LoRA dropout during training

The configuration uses:

```python
LORA_DROPOUT = 0.1
```

During training, 10% of values entering a LoRA path are randomly set aside for
that pass. This changes which LoRA inputs are active from one pass to another.

LoRA dropout is disabled during evaluation and inference.

## 35. Finish one physical batch

After one physical batch of 16 examples:

```text
Forward pass -> loss
Backward pass -> LoRA gradients
```

The experiment uses four gradient-accumulation steps. The optimizer does not
update LoRA yet after the first physical batch.

## 36. Accumulate four physical batches

The sequence is:

```text
Physical batch 1: calculate and keep LoRA gradients
Physical batch 2: calculate and add LoRA gradients
Physical batch 3: calculate and add LoRA gradients
Physical batch 4: calculate and add LoRA gradients
```

Each batch contains 16 examples, so one optimizer update receives information
from approximately:

```text
16 * 4 = 64 examples
```

This is the effective batch size.

Only one physical batch's forward-pass activations need to exist at a time.
Accumulated gradients remain until the optimizer update.

## 37. Calculate and clip the gradient norm

The trainer calculates one summary magnitude for all trainable gradients. This
is the gradient norm.

The configuration uses:

```python
max_grad_norm = 0.3
```

If the combined gradient norm is too large, the gradients are scaled down
before the update. This is gradient clipping.

The logged gradient norm can show the value measured before clipping, so it
can be larger than `0.3`.

## 38. Update LoRA with AdamW

After four accumulated physical batches, AdamW updates every trainable LoRA
parameter.

A very simplified update is:

```text
new LoRA parameter = old LoRA parameter - learning rate * processed gradient
```

AdamW also uses moving histories of gradients and squared gradients, plus
weight decay. Therefore, the real update contains more calculations than this
simplified expression.

Only LoRA matrices are updated. Original Qwen matrices remain unchanged.

## 39. Increase global step

After the optimizer update:

```text
global_step increases by 1
```

With 1,000 examples and an effective batch size of 64:

```text
1,000 / 64 = approximately 16 global steps
```

This produced the `0/16` to `16/16` progress bar.

## 40. Update the learning rate

The learning-rate scheduler changes the learning rate after optimizer updates.

The experiment used a warmup followed by cosine decay:

```text
start small
-> reach the maximum learning rate
-> gradually decrease toward zero
```

The logged learning rates decreased during the short run because the complete
schedule was compressed into only 16 optimizer updates.

## 41. Clear gradients

After the optimizer update, accumulated gradients are cleared. The next group
of four physical batches starts with new gradients.

The cycle repeats:

```text
four forward/backward passes
-> clip gradients
-> update LoRA
-> update learning rate
-> clear gradients
-> increase global step
```

## 42. Complete one epoch

One epoch is complete after all selected training examples have been
processed once.

For the first experiment:

```text
Training examples: 1,000
Epochs: 1
Global steps: 16
```

If three epochs are configured, the same training split is processed three
times, with LoRA continuing from its parameters learned in earlier epochs.

## 43. Run validation

Validation uses a separate dataset split that does not update LoRA.

During validation:

```text
LoRA dropout is disabled
gradients are not retained
the optimizer does not run
LoRA parameters do not change
```

The trainer calculates validation loss, token accuracy, and entropy. These
metrics show how the current adapter performs on examples it did not train on.

## 44. Save the adapter

After training, PEFT saves files such as:

```text
adapter_config.json
adapter_model.safetensors
```

The adapter file contains the trained LoRA matrices. It does not contain a
complete second copy of all Qwen parameters.

That is why the trained adapter was about 29.5 MB while the base model contains
billions of parameters.

## 45. Load the trained adapter later

When Transformers receives the adapter repository, it finds
`adapter_config.json`. That configuration names the original base model.

Transformers then:

1. Loads Qwen/Qwen2.5-3B-Instruct.
2. Loads the trained LoRA matrices.
3. Attaches them to q_proj, k_proj, v_proj, and o_proj.
4. Activates the adapter.

The base and adapter weights remain logically separate.

## 46. Send a new prompt after training

During inference, only the prompt is supplied:

```text
What does this product cost?

Steel door knob with mounting hardware.

Price is $
```

The model generates the completion itself.

For every input and generated token, the message passes through all 36 layers.
In every targeted projection, Qwen calculates:

```text
final projection result = original result + trained LoRA adjustment
```

LoRA is active inside the projections. It is not applied once before the base
model or once after the base model.

## 47. Generate one token

After the final layer, Qwen produces 151,936 next-token scores. Generation
selects a token according to its generation settings.

For deterministic generation with `do_sample=False`, the highest-scoring
allowed token is selected.

The selected token is appended to the sequence. Qwen then calculates the next
token using the prompt and the token it just generated.

This repeats until:

```text
an EOS token is generated
or
max_new_tokens is reached
```

## 48. The complete training flow

```text
Prompt and completion
-> tokenize
-> build completion labels
-> truncate and pad
-> embedding lookup
-> Transformer layer 1
   -> attention with q/k/v/o LoRA adjustments
   -> MLP
-> repeat through layer 36
-> vocabulary scores
-> completion-token loss
-> backpropagation
-> LoRA gradients
-> accumulate four physical batches
-> gradient clipping
-> AdamW updates LoRA
-> learning-rate scheduler updates
-> repeat until epoch completes
-> validation
-> save adapter
```

## 49. The central QLoRA facts

```text
The base Qwen weights are loaded in compressed 4-bit form.
The base Qwen weights remain frozen.
The LoRA matrices are small and trainable.
LoRA adjustments run inside selected projections.
Loss measures completion-token predictions.
Backpropagation calculates gradients for LoRA.
The optimizer changes LoRA parameters.
The saved adapter is used together with the original Qwen model.
```

The central projection formula is:

```text
final output = W(x) + (alpha / rank) * B(A(x))
```

Where:

```text
W -> frozen original Qwen matrix
A -> trained first LoRA matrix
B -> trained second LoRA matrix
x -> projection input
```
