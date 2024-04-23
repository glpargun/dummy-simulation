import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.example_libraries import optimizers

class NeuralNetworkModel:
    def __init__(self, layer_sizes, step_size=0.01):
        self.step_size = step_size
        self.params = self.init_network_params(layer_sizes)
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(self.step_size)
        self.opt_state = self.opt_init(self.params)
        self.loss_grad = jit(grad(self.loss))

    def penalty_loss(self, params, x, y, b):
        y_pred = self.predict(params, x)
        sigma = jnp.abs(y_pred - y)
        return jnp.exp(-b / sigma)

    def init_network_params(self, layer_sizes, rng_key=jax.random.PRNGKey(0)):
        keys = jax.random.split(rng_key, len(layer_sizes) - 1)
        params = []
        for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
            w = jax.random.normal(k, (m, n)) * jnp.sqrt(2.0 / m)
            b = jnp.zeros(n)
            params.append((w, b))
        return params

    def predict(self, params, x):
        activations = x
        for w, b in params[:-1]:
            activations = jnp.tanh(jnp.dot(activations, w) + b)
        final_w, final_b = params[-1]
        return jnp.dot(activations, final_w) + final_b

    def loss(self, params, x, y):
        pred = self.predict(params, x)
        return jnp.mean((pred - y) ** 2)

    def update(self, epoch, opt_state, x, y):
        params = self.get_params(opt_state)
        gradients = self.loss_grad(params, x, y)
        return self.opt_update(epoch, gradients, opt_state)

    def train_network(self, x, y, epochs=1000):
        for epoch in range(epochs):
            self.opt_state = self.update(epoch, self.opt_state, x, y)
            if epoch % 10 == 0:
                params = self.get_params(self.opt_state)
                print(f"Epoch {epoch}: loss {self.loss(params, x, y)}")
        return self.get_params(self.opt_state)
