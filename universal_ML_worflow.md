# The Universal Workflow of Machine Learning

## 1. Define the problem and assemble a dataset

- What will the input data be? What are you trying to predict?
- What type of problem? Binary classification? Scalar regression? Multiclass classification? Vector regression? Multiclass, multilabel classification?

At this point everything rests on two hypotheses:

- Outputs can be predicted given the inputs.
- Available data are sufficiently informative to learn the relationship between inputs and outputs.

# 2. Choose a measure of success

How will you decide that a model is successful? Simple accuracy or ROC AUC may be sufficient on a balanced class problem. For imbalanced classes may need to consider precision, recall, F1.

# 3. Decide on an evaluation protocol

Three options:

- Hold-out validation set;
- K-fold validation;
- Iterated k-fold validation.

# 4. Prepare the data

- Format data as tensors.
- Scale data to have small values (e.g. by dividing each value by the range or maximum).
- Normalise features that have different ranges.
- Consider feature engineering (especially if there isn't much data).

# 5. Build a model that is better than a baseline

Achieve _statistical power_ with a model that performs better than some minimal baseline. For example: in the MNIST problem doing better than 0.1 accuracy would have some power.

Three choices to develop this first working model:

1. _Last-layer activation_: Constrains the model's output appropriately, e.g. with `sigmoid` on the last layer to generate a single probability of 'success'.
2. _Loss function_: appropriate to the type of problem, e.g. `categorical_crossentropy` for a multiclass problem.
3. _Optimisation configuration_: choosing the optimiser and its learning rate. (Usually safe to start with `rmsprop` and its default learning rate.)

NB. Cannot always match the loss function and metric: loss function must be differentiable, which isn't true of many metrics. 

Suggestions for items 1 and 2 in this table.

| Problem type.              | Output activation | Loss function              |
| :------------------------- | :---------------- | :------------------------- |
| Binary classification      | `sigmoid`         | `binary_crossentropy`      |
| Multiclass, single-label   | `softmax`         | `categorical_crossentropy` |
| Multiclass, multilabel     | `sigmoid`         | `binary_crossentropy`      |
| Regression to any value    | None              | `mse`                      |
| Regression to value [0-1] | `sigmoid`         | `mse`/`binary_crossentropy` |

# 6. Scaling up: developing a model that overfits

End goal is to balance between optimisation on the training data, and generalising to new data. To find that border between them, have to cross it into overfitting. To figure out how big a model you’ll need, you must develop a model that overfits.
This is fairly easy:
1. Add layers.
2. Make the layers bigger.
3. Train for more epochs.

Once the performance on validation data starts to degrade, overfitting has started.

# 7. Regularising, tuning hyperparameters

Repeat the process of changing the model, training, evaluating (on validation only), then making more changes etc. until the model is as good as possible. Some of the steps to try:

- Add dropout.
- Try different architectures: add or remove layers.
- Add L1 and/or L2 regularization.
- Try different hyperparameters (such as the number of units per layer or the
learning rate of the optimizer) to find the optimal configuration.
- Optionally, iterate on feature engineering: add new features, or remove fea-
tures that don’t seem to be informative.

However this process comes with a danger: every time you change the model based on results from validation, information leaks between the partitions. Done enough times this will create a model that has learned on the validation set, and will therefore not generalise.

Once the model is satisfactory, can then train a final production model on all the training and validation data, and evaluate it once on the test set. If it performs significantly worse than on the validation set this may mean either that your validation procedure wasn’t reliable after all, or that you began overfitting to the validation data while tuning the parameters of the model. In this case, you may want to switch to a more reliable evaluation protocol (such as iterated K-fold validation).