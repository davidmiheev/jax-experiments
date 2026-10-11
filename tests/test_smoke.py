"""The toolchain works: jit, grad, jvp, vmap, scan, and the Flax NNX and Optax APIs the plan relies on."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx


def test_jit_grad_matches_analytic():
    f = jax.jit(jax.grad(lambda x: jnp.sum(jnp.tanh(x) ** 2)))
    x = jnp.arange(3.0)
    expected = 2 * jnp.tanh(x) * (1 - jnp.tanh(x) ** 2)
    assert jnp.allclose(f(x), expected, atol=1e-6)


def test_forward_and_reverse_mode_agree():
    def f(x):
        return jnp.sin(x) * x**2

    x, v = jnp.array([0.3, -1.2, 2.0]), jnp.array([1.0, 0.5, -2.0])
    _, jvp_out = jax.jvp(f, (x,), (v,))
    jacobian = jax.jacrev(f)(x)
    assert jnp.allclose(jvp_out, jacobian @ v, atol=1e-6)


def test_vmap_per_example_gradients():
    def loss(w, x, y):
        return (x @ w - y) ** 2

    w = jnp.array([1.0, -2.0])
    xs, ys = jnp.ones((4, 2)), jnp.arange(4.0)
    per_example = jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))(w, xs, ys)
    assert per_example.shape == (4, 2)
    assert jnp.allclose(per_example.mean(0), jax.grad(lambda w: jnp.mean(jax.vmap(loss, (None, 0, 0))(w, xs, ys)))(w))


def test_scan_matches_python_loop():
    def step(h, x):
        h = jnp.tanh(0.5 * h + x)
        return h, h

    xs = jnp.linspace(-1, 1, 7)
    _, scanned = jax.lax.scan(step, 0.0, xs)
    h, looped = 0.0, []
    for x in xs:
        h, out = step(h, x)
        looped.append(out)
    assert jnp.allclose(scanned, jnp.stack(looped), atol=1e-6)


def test_nnx_optax_training_step_reduces_loss():
    model = nnx.Linear(3, 1, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-1), wrt=nnx.Param)
    x = jnp.ones((8, 3))
    y = jnp.full((8, 1), 2.0)

    @nnx.jit
    def train_step(model, optimizer):
        loss, grads = nnx.value_and_grad(lambda m: jnp.mean((m(x) - y) ** 2))(model)
        optimizer.update(model, grads)
        return loss

    first = train_step(model, optimizer)
    for _ in range(20):
        last = train_step(model, optimizer)
    assert last < first
