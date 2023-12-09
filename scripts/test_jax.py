# The following code snippet will be run on all TPU hosts
import numpy as np
import jax
from jax.experimental import mesh_utils
import numpy as np
Mesh, NamedSharding = jax.sharding.Mesh, jax.sharding.NamedSharding
P, with_sharding_constraint = jax.sharding.PartitionSpec, jax.lax.with_sharding_constraint


def tree_broadcast(prefix, target):
  def _broadcast(leaf, subtree):
    return jax.tree_map(lambda _: leaf, subtree)
  return jax.tree_map(_broadcast, prefix, target)


def reshard(tree, shardings):
  def _make_global_arr(x, shard, shape):
    # Avoid unnecessary copies and transfers:
    if hasattr(x, "sharding") and x.sharding.is_equivalent_to(shard, len(shape)):  # pylint: disable=line-too-long
      return x
    if not getattr(x, "is_fully_addressable", True):
      raise RuntimeError("Trying to reshard a non-fully-addressable array. "
                         "Please see the doc-comment for detailed explanation.")
    x = jax.device_get(x)  # Might be on local devices.
    xs = [jax.device_put(x[s], device=d)
          for d, s in shard.addressable_devices_indices_map(shape).items()]
    return jax.make_array_from_single_device_arrays(shape, shard, xs)

  shapes = jax.tree_map(np.shape, tree)
  shardings = tree_broadcast(shardings, tree)
  return jax.tree_map(_make_global_arr, tree, shardings, shapes)


# The total number of TPU cores in the Pod
device_count = jax.device_count()

# The number of TPU cores attached to this host
local_device_count = jax.local_device_count()

mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), axis_names=('data',))
A = np.ones((128 * 64, 128 * 100))
B = np.ones((128 * 100, 256))
shardings = (NamedSharding(mesh, P('data', None)), NamedSharding(mesh, P('data', None)))
A, B = reshard((A, B), shardings)

@jax.jit
def op(A, B):
    return A @ B

result = op(A, B)

# Print from a single host to avoid duplicated output
if jax.process_index() == 0:
    print('global device count:', jax.device_count())
    print('local device count:', jax.local_device_count())
    jax.debug.visualize_array_sharding(A)
    jax.debug.visualize_array_sharding(B)
    jax.debug.visualize_array_sharding(result)