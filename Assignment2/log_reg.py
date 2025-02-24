import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("TF version:", tf.__version__)
print("Eager execution:", tf.executing_eagerly())

from tensorflow.keras.datasets import fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = (train_images / 255.0).astype(np.float32)
test_images = (test_images / 255.0).astype(np.float32)

val_split = 0.1
num_val = int(train_images.shape[0] * val_split)
val_images = train_images[:num_val]
val_labels = train_labels[:num_val]
train_images_split = train_images[num_val:]
train_labels_split = train_labels[num_val:]

class LogisticRegressionModel(tf.Module):
    def __init__(self):
        super().__init__().
        self.W = tf.Variable(tf.zeros([28 * 28, 10]), name="weights")
        self.b = tf.Variable(tf.zeros([10]), name="biases")

    def __call__(self, x):
       
        x = tf.reshape(x, [-1, 28 * 28])
        logits = tf.matmul(x, self.W) + self.b

        return tf.nn.softmax(logits)


def compute_loss(model, images, labels, lambda_reg=0.0):
    predictions = model(images)

    ce_loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(labels, predictions))

    l2_loss = tf.nn.l2_loss(model.W)
    return ce_loss + lambda_reg * l2_loss


def train_step(model, images, labels, optimizer, lambda_reg=0.0):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, images, labels, lambda_reg)
    grads = tape.gradient(loss, [model.W, model.b])
    optimizer.apply_gradients(zip(grads, [model.W, model.b]))
    return loss


def train_model(optimizer, lambda_reg=0.0, num_epochs=10, batch_size=128):

    model = LogisticRegressionModel()

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images_split, train_labels_split))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_dataset = val_dataset.batch(batch_size)
    
  
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(1, num_epochs + 1):
        train_acc_metric.reset_state()
        val_acc_metric.reset_state()
        epoch_losses = []
        .
        for batch_images, batch_labels in train_dataset:
            loss = train_step(model, batch_images, batch_labels, optimizer, lambda_reg)
            epoch_losses.append(loss.numpy())
            predictions = model(batch_images)
            train_acc_metric.update_state(batch_labels, predictions)
    
        train_loss = np.mean(epoch_losses)
        train_accuracy = train_acc_metric.result().numpy()

       
        val_losses = []
        for batch_images, batch_labels in val_dataset:
            loss = compute_loss(model, batch_images, batch_labels, lambda_reg)
            val_losses.append(loss.numpy())
            predictions = model(batch_images)
            val_acc_metric.update_state(batch_labels, predictions)
        val_loss = np.mean(val_losses)
        val_accuracy = val_acc_metric.result().numpy()

       
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(f"Epoch {epoch:02d}: Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")
    return model, history


optimizers_to_try = {
    "SGD": tf.optimizers.SGD(learning_rate=0.5),
    "Adam": tf.optimizers.Adam(learning_rate=0.001),
    "RMSprop": tf.optimizers.RMSprop(learning_rate=0.001)
}


lambda_reg = 0.001
num_epochs = 10
history_dict = {}

for opt_name, opt in optimizers_to_try.items():
    print(f"\nTraining with {opt_name} optimizer (lambda_reg={lambda_reg})")

    _, history = train_model(opt, lambda_reg=lambda_reg, num_epochs=num_epochs, batch_size=128)
    history_dict[opt_name] = history


epochs = range(1, num_epochs + 1)
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

for opt_name, history in history_dict.items():
    axs[0].plot(epochs, history["train_loss"], label=f"{opt_name} Train")
    axs[0].plot(epochs, history["val_loss"], '--', label=f"{opt_name} Val")
axs[0].set_title("Loss over Epochs")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend()
axs[0].grid(True)

for opt_name, history in history_dict.items():
    axs[1].plot(epochs, history["train_accuracy"], label=f"{opt_name} Train")
    axs[1].plot(epochs, history["val_accuracy"], '--', label=f"{opt_name} Val")
axs[1].set_title("Accuracy over Epochs")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()


