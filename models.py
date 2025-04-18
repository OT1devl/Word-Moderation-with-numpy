import numpy as np
import time
import pickle

from functions import ReLU, sigmoid, BinaryCrossEntropyLoss
from optimizers import Optimizer
from settings import DTYPE, model_path

class Embedding:
    def __init__(self, vocab_size, embedding_dim, trainable=True, mask_zero=False):
        self.trainable = trainable
        self.mask_zero = mask_zero
        self.W = np.random.randn(vocab_size, embedding_dim).astype(DTYPE) * np.sqrt(2/vocab_size)

    def forward(self, x):
        self.inputs = x
        self.mask = (x != 0) if self.mask_zero else None
        self.outputs = self.W[x]
        return self.outputs

    def backward(self, dout):
        if not self.trainable:
            return None
        
        dW = np.zeros_like(self.W)
        np.add.at(dW, self.inputs, dout)

        if self.mask_zero:
            dW[0] = 0

        return dW

class ModerationNetwork:
    def __init__(self, vocab_size: int, embedding_dim: int, neurons: int=64):
        self.embedding = Embedding(vocab_size, embedding_dim, trainable=True, mask_zero=True)

        self.W0 = np.random.randn(embedding_dim, neurons).astype(DTYPE) * np.sqrt(2/embedding_dim)
        self.b0 = np.zeros((1, neurons)).astype(DTYPE)

        self.W1 = np.random.randn(neurons, neurons//2).astype(DTYPE) * np.sqrt(2/neurons)
        self.b1 = np.zeros((1, neurons//2)).astype(DTYPE)

        self.W2 = np.random.randn(neurons//2, 1).astype(DTYPE) * np.sqrt(2/(neurons//2))
        self.b2 = np.zeros((1, 1)).astype(DTYPE)

        self.params = [self.embedding.W,
                       self.W0, self.b0,
                       self.W1, self.b1,
                       self.W2, self.b2]
        
        self.n_params = len(self.params)

    def compile(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.optimizer.init_cache(self.n_params)

    def update_params(self, gradients):
        self.optimizer.prev_update()
        for i, grad in enumerate(gradients):
            self.params[i] -= self.optimizer.update_params(grad, i)
        self.optimizer.step()

    def forward(self, x):
        self.embed = self.embedding.forward(x)

        if self.embedding.mask_zero:
            mask = self.embedding.mask
            masked = self.embed * mask[:, :, None]
            summed = np.sum(masked, axis=1)
            counts = np.sum(mask, axis=1)
            counts = np.where(counts == 0, 1, counts)
            self.pooled = summed / counts[:, None]
        else:
            self.pooled = np.mean(self.embed, axis=1)

        self.z0 = self.pooled @ self.W0 + self.b0
        self.a0 = ReLU(self.z0)

        self.z1 = self.a0 @ self.W1 + self.b1
        self.a1 = ReLU(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def backward(self, y, learn=True):
        m = y.shape[0]
        dL = BinaryCrossEntropyLoss(y, self.a2, derv=True) * sigmoid(self.a2, derv=True)

        dW2 = self.a1.T @ dL / m
        db2 = np.sum(dL, axis=0, keepdims=True) / m

        dz1 = dL @ self.W2.T * ReLU(self.z1, derv=True)

        dW1 = self.a0.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        dz0 = dz1 @ self.W1.T * ReLU(self.z0, derv=True)

        dW0 = self.pooled.T @ dz0 / m
        db0 = np.sum(dz0, axis=0, keepdims=True)

        dpooled = dz0 @ self.W0.T

        seq_len = self.embed.shape[1]
        dembed = np.repeat(dpooled[:, np.newaxis, :], seq_len, axis=1)

        if self.embedding.mask_zero:
            mask = self.embedding.mask[..., np.newaxis]
            counts = np.sum(mask, axis=1, keepdims=True)
            counts[counts == 0] = 1
            dembed = dembed * mask / counts
        else:
            dembed /= seq_len

        dW_embed = self.embedding.backward(dembed)

        if learn:
            self.update_params((dW_embed, dW0, db0, dW1, db1, dW2, db2))

    def train(self, x, y, epochs=10, batch_size=32, verbose=True, print_every=2):
        total_batchs = np.ceil(x.shape[0] / batch_size).astype(int)

        for ep in range(1, epochs+1):

            total_loss = 0.0
            start = time.time()

            for b in range(0, x.shape[0], batch_size):
                x_batch = x[b:b+batch_size]
                y_batch = y[b:b+batch_size]
                total_loss += BinaryCrossEntropyLoss(y_batch, self.forward(x_batch))
                self.backward(y_batch, learn=True)
            
            avg_loss = total_loss / total_batchs

            if verbose and ep % print_every == 0:
                avg_time = time.time() - start
                print(f'Epoch: [{ep}/{epochs}] (time: {avg_time:.2f} secs)> Loss: {avg_loss:.4f}')

    def save(self, path: str=model_path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str=model_path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
            
        return model