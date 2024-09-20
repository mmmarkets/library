import jax
import jax.numpy as jnp

from model import Model
from differentials import expression, forward

# let u be approximated by u^

# u = Model() # u is fit by the expression

dx = lambda u: jax.grad(u, argnums=1)
dt = lambda u: jax.grad(u, argnums=2)

heat = expression(
		lambda u: lambda x, t: dt(u)(x, t) + dx(dx(u))(x, t),
		var = ("x", "t"),
		x = jnp.linspace(-1, 1, num=100),  # for x domain
		t = jnp.linspace(0, 1, num=100)    # for t domain
		)

u_model, params = heat.u()
x = jnp.array([float(0), float(1)])
u = lambda x, t: u_model.apply(params, jnp.array([x, t]))
print(u(0, 1))
ux = dx(u)
print(type(ux))
print(ux(0,1))
