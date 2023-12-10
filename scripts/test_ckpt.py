import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
import orbax.checkpoint as ocp
Mesh, NamedSharding = jax.sharding.Mesh, jax.sharding.NamedSharding
P, with_sharding_constraint = jax.sharding.PartitionSpec, jax.lax.with_sharding_constraint

jax.distributed.initialize()

options = ocp.CheckpointManagerOptions(
    max_to_keep=1, save_interval_steps=100)
mngr = ocp.CheckpointManager(
    "gs://train_out/test",
    ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
    options=options)

mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), axis_names=('data',))
shardings = NamedSharding(mesh, P('data', None))
@jax.jit
def init():
    x = jnp.ones((128 * 64, 128 * 100))
    return jax.with_sharding_constraint(x, shardings)
A = init()
mngr.save(0, A)