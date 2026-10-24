# How LoRA Works

This lesson explains what happens when you train and use a LoRA adapter.


1. THE BASE MODEL
-----------------

Qwen is a pretrained neural network containing billions of stored numbers.
These stored numbers are called parameters or weights.

Your base model is:

    Qwen/Qwen2.5-3B-Instruct

The model contains 36 Transformer layers. Inside the attention section of
each layer are four linear projections:

    q_proj
    k_proj
    v_proj
    o_proj

Each projection contains a large weight matrix. We can call one of these
original weight matrices W.


2. WHAT FULL FINE-TUNING WOULD DO
---------------------------------

During full fine-tuning, training can change the original weight matrices
inside the base model.

For a model with billions of parameters, changing all those parameters needs
a large amount of memory for:

    weights
    gradients
    optimizer state
    intermediate calculations


3. WHAT LoRA DOES
-----------------

LoRA keeps the original weight matrix W frozen.

Frozen means:

    W is still used in calculations.
    Training does not change the numbers stored in W.

LoRA adds two smaller trainable matrices to a selected projection. We can call
them A and B.

During training:

    Original matrix W  -> frozen
    LoRA matrix A      -> trained
    LoRA matrix B      -> trained

The calculation becomes:

    final output = W(x) + scale * B(A(x))

Here:

    x               is the input entering the projection
    W(x)            is the original Qwen result
    B(A(x))         is the LoRA adjustment
    scale           controls the strength of the adjustment
    final output    is the result passed to the next calculation

LoRA does not replace the original calculation. It adds an adjustment to it.


4. WHAT THE LoRA RANK MEANS
---------------------------

The LoRA rank is written as r.

Your configuration used:

    r = 32
    alpha = 64

Suppose an original projection receives 2,048 input numbers and produces
2,048 output numbers.

Instead of training another complete 2,048 by 2,048 matrix, LoRA creates:

    Matrix A: 2,048 inputs -> 32 numbers
    Matrix B: 32 numbers   -> 2,048 outputs

The value 32 is the LoRA rank.

A smaller rank creates fewer trainable parameters. A larger rank creates more
trainable parameters and gives the adapter more capacity to learn changes.


5. WHERE LoRA WAS ADDED IN YOUR MODEL
-------------------------------------

Your configuration targeted:

    q_proj
    k_proj
    v_proj
    o_proj

These are four projections inside the attention section of every Transformer
layer.

Qwen2.5-3B-Instruct has 36 Transformer layers, so LoRA was attached to:

    36 layers * 4 projections = 144 projection modules


6. WHAT THE FOUR PROJECTIONS DO
-------------------------------

q_proj creates query numbers.

    Query numbers participate in deciding what information the current token
    should select from available tokens.

k_proj creates key numbers.

    Query numbers are compared with key numbers to calculate attention scores.

v_proj creates value numbers.

    Value numbers contain the information that attention combines using the
    calculated attention weights.

o_proj creates the attention output.

    It converts the combined attention information back into the model's
    normal hidden size for the rest of the Transformer layer.


7. WHAT GETS SAVED AFTER TRAINING
---------------------------------

Your training saved a LoRA adapter. The important files include:

    adapter_config.json
    adapter_model.safetensors

adapter_config.json describes:

    the original base model
    the LoRA rank
    the LoRA alpha
    the targeted projection modules

adapter_model.safetensors contains the trained LoRA A and B matrices.

The adapter file is much smaller than the full Qwen model because it does not
contain another copy of all Qwen weights.


8. WHAT HAPPENS WHEN THE TRAINED MODEL IS LOADED
------------------------------------------------

When you load this repository:

    amitdeshmukh/price-2026-10-24_17.16.25-lite

Transformers finds adapter_config.json and performs these steps:

    1. Read which base model the adapter requires.
    2. Load Qwen/Qwen2.5-3B-Instruct.
    3. Load the trained LoRA matrices.
    4. Attach them to q_proj, k_proj, v_proj and o_proj.
    5. Activate the adapter for inference.

The original Qwen weights and the LoRA weights remain separate.


9. WHEN LoRA KICKS IN
---------------------

Suppose you send this message:

    What does this steel door knob cost?

The sequence is:

    1. The tokenizer converts the text into token IDs.
    2. Qwen converts every token ID into a numeric representation.
    3. Those numbers enter Transformer layer 1.
    4. When they reach q_proj, both Qwen and LoRA calculate results.
    5. The LoRA adjustment is added to the original q_proj result.
    6. The same happens in k_proj, v_proj and o_proj.
    7. The process repeats in the remaining Transformer layers.
    8. The final model scores possible next tokens.
    9. Generation selects the next token and repeats the process.

LoRA is not a separate calculation performed only before or after Qwen.
It runs inside every targeted projection while the message passes through the
model.

When the adapter is enabled, it runs for every input token and every generated
token.


10. A SMALL NUMERICAL EXAMPLE
-----------------------------

We will use a tiny projection with:

    3 input numbers
    2 output numbers
    LoRA rank = 1

The real Qwen model uses much larger dimensions, but the calculation is the
same.

Input:

    x = [2, 1, 3]

Original Qwen weight matrix:

    W = [
        [0.5,  0.2, -0.1],
        [0.1, -0.4,  0.3]
    ]

Calculate the first original output:

    2 * 0.5 + 1 * 0.2 + 3 * (-0.1)
    = 1.0 + 0.2 - 0.3
    = 0.9

Calculate the second original output:

    2 * 0.1 + 1 * (-0.4) + 3 * 0.3
    = 0.2 - 0.4 + 0.9
    = 0.7

The original Qwen result is:

    W(x) = [0.9, 0.7]


11. CALCULATE THE FIRST LoRA MATRIX
-----------------------------------

The first LoRA matrix is:

    A = [
        [0.2, 0.1, -0.1]
    ]

It converts three input numbers into one number:

    2 * 0.2 + 1 * 0.1 + 3 * (-0.1)
    = 0.4 + 0.1 - 0.3
    = 0.2

Therefore:

    A(x) = [0.2]


12. CALCULATE THE SECOND LoRA MATRIX
------------------------------------

The second LoRA matrix is:

    B = [
        [ 0.5],
        [-0.25]
    ]

It converts one number into two numbers:

    First output:  0.2 * 0.5   =  0.1
    Second output: 0.2 * -0.25 = -0.05

Therefore:

    B(A(x)) = [0.1, -0.05]


13. APPLY LoRA SCALING
----------------------

Ordinary LoRA scaling is calculated using:

    scale = alpha / rank

Your configuration used:

    alpha = 64
    rank  = 32

Therefore:

    scale = 64 / 32 = 2

Apply the scale to our LoRA result:

    [0.1, -0.05] * 2
    = [0.2, -0.1]

The final LoRA adjustment is:

    [0.2, -0.1]


14. ADD LoRA TO THE ORIGINAL RESULT
-----------------------------------

Original Qwen result:

    [0.9, 0.7]

LoRA adjustment:

    [0.2, -0.1]

Final result:

    [0.9, 0.7] + [0.2, -0.1]
    = [1.1, 0.6]

Without LoRA, the projection returns:

    [0.9, 0.7]

With LoRA, the projection returns:

    [1.1, 0.6]


15. THE CORE IDEA
-----------------

During training:

    Keep Qwen's original W matrices frozen.
    Train the smaller A and B matrices.

During inference:

    Use the original Qwen calculation.
    Calculate the trained LoRA adjustment.
    Add both results.

The essential formula is:

    final output = W(x) + (alpha / rank) * B(A(x))

This happens inside every targeted projection while tokens pass through the
model.
