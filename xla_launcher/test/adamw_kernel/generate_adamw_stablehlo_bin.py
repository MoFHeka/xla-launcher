import sys

import numpy as np
import os
import inspect
import yaml

import jax
import jax.numpy as jnp
from jax import export
# Add MLIR Python API imports
from jax._src.interpreters import mlir as mlir_jax
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import func as func_dialect
from jax.extend.mlir.dialects import stablehlo as stablehlo_dialect


def adamw_kernel(params, grads, m, v, step, lr, b1, b2, eps, weight_decay):
    """
  JAX implementation of a single AdamW optimizer step.

  Args:
    params: A pytree of parameters.
    grads: A pytree of gradients, same structure as params.
    m: A pytree for the first moment estimates, same structure as params.
    v: A pytree for the second moment estimates, same structure as params.
    step: A scalar integer tensor for the current step.
    lr: Learning rate.
    b1: Exponential decay rate for the first moment estimates.
    b2: Exponential decay rate for the second moment estimates.
    eps: A small constant for numerical stability.
    weight_decay: Weight decay coefficient.

  Returns:
    A tuple of (new_params, new_m, new_v, new_step).
  """
    step_new = step + 1

    # Update biased first moment estimate.
    m_new = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, m,
                                   grads)
    # Update biased second raw moment estimate.
    v_new = jax.tree_util.tree_map(
        lambda v, g: b2 * v + (1 - b2) * jnp.square(g), v, grads)

    # Compute bias-corrected first moment estimate.
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - b1**step_new), m_new)
    # Compute bias-corrected second raw moment estimate.
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - b2**step_new), v_new)

    # decoupled weight decay and parameter update
    update = jax.tree_util.tree_map(
        lambda m, v, p: lr * (m / (jnp.sqrt(v) + eps) + weight_decay * p),
        m_hat, v_hat, params)

    params_new = jax.tree_util.tree_map(lambda p, u: p - u, params, update)

    return params_new, m_new, v_new, step_new


def get_signature_list(names, specs, dynamic_dims):
    """
    Generates a list of signatures for inputs or outputs.

    Args:
        names: A list of names for the arguments.
        specs: A dictionary mapping names to their ShapeDtypeStruct.

    Returns:
        A list of dictionaries, each representing an argument's signature.
    """
    signature = []
    for name in names:
        spec = specs[name]
        shape = [
            dynamic_dims[d] if d in dynamic_dims else str(d)
            for d in spec.shape
        ]
        dtype = np.dtype(spec.dtype).name
        signature.append({'name': name, 'shape': shape, 'dtype': dtype})
    return signature


def main():
    """
  Main function to generate StableHLO and test data for AdamW.
  """
    if len(sys.argv) < 2:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        script_dir = sys.argv[1]

    # 1. Define hyperparameters
    lr = np.float32(1e-3)
    b1 = np.float32(0.9)
    b2 = np.float32(0.999)
    eps = np.float32(1e-8)
    weight_decay = np.float32(1e-2)

    # 2. Create sample data for a single parameter, gradient, and optimizer state.
    # Use fixed data instead of random for deterministic testing.
    param_shape = (128, 256)

    # Create deterministic initial values for params and grads.
    # Using np.arange and reshaping to get varied but deterministic values.
    size = int(np.prod(param_shape))
    params = jnp.array(
        np.arange(0, size, dtype=np.float32).reshape(param_shape) / size)
    grads = jnp.array(
        np.arange(size, 2 * size, dtype=np.float32).reshape(param_shape) /
        size)

    m = jax.random.uniform(jax.random.PRNGKey(0), param_shape, dtype=jnp.float32)
    v = jax.random.uniform(jax.random.PRNGKey(1), param_shape, dtype=jnp.float32)
    step = jnp.array(0, dtype=jnp.int32)

    arg_names = list(inspect.signature(adamw_kernel).parameters.keys())
    arg_names_to_donate = ["params", "m", "v"]
    donate_argnums = tuple(
        [i for i, name in enumerate(arg_names) if name in arg_names_to_donate])

    # 3. Use jax.export to get the StableHLO representation with dynamic shapes.
    # We define symbolic shapes for the tensor arguments.
    symbolic_dims = export.symbolic_shape(
        "batch_size, emb_dim")  # Dynamic shapes name
    e_params = jax.ShapeDtypeStruct(symbolic_dims, jnp.float32)
    e_grads = jax.ShapeDtypeStruct(symbolic_dims, jnp.float32)
    e_m = jax.ShapeDtypeStruct(symbolic_dims, jnp.float32)
    e_v = jax.ShapeDtypeStruct(symbolic_dims, jnp.float32)

    # Scalars remain as they are, jax.export handles them correctly.
    e_step = jax.ShapeDtypeStruct((), jnp.int32)
    e_lr = jax.ShapeDtypeStruct((), np.float32)
    e_b1 = jax.ShapeDtypeStruct((), np.float32)
    e_b2 = jax.ShapeDtypeStruct((), np.float32)
    e_eps = jax.ShapeDtypeStruct((), np.float32)
    e_weight_decay = jax.ShapeDtypeStruct((), np.float32)

    jitted_kernel = jax.jit(adamw_kernel, donate_argnums=donate_argnums)

    exp = export.export(jitted_kernel)  # No Platforms
    exp_compile = exp(e_params, e_grads, e_m, e_v, e_step, e_lr, e_b1, e_b2,
                      e_eps, e_weight_decay)
    stablehlo_module_str = exp_compile.mlir_module()

    # Replace SSA names in the MLIR module for better readability.
    # We iterate backwards to prevent conflicts like replacing "%arg1" in "%arg10".
    for i, name in reversed(list(enumerate(arg_names))):
        stablehlo_module_str = stablehlo_module_str.replace(
            f'%arg{i}', f'%{name}')

    # To add meaningful location metadata, we perform a targeted string
    # replacement. This is more robust than parsing the IR, which is
    # restricted by the JAX bindings. We start replacement from the @main
    # function to avoid altering other parts of the module.
    main_func_anchor = "func.func public @main"
    anchor_pos = stablehlo_module_str.find(main_func_anchor)
    if anchor_pos != -1:
        pre_func_str = stablehlo_module_str[:anchor_pos]
        func_and_body_str = stablehlo_module_str[anchor_pos:]
        for name in arg_names:
            func_and_body_str = func_and_body_str.replace(
                'loc(unknown)', f'loc("{name}")', 1)
        stablehlo_module_str = pre_func_str + func_and_body_str

    # # Use MLIR Python API to rename arguments and update locations for readability.
    # with mlir_jax.JaxIrContext() as ctx:
    #     # Load dialects that are used in the module.
    #     ctx.append_dialect_registry(mlir_jax.upstream_dialects)
    #     stablehlo_dialect.register_dialect(ctx)
    #     ctx.load_all_available_dialects()
    #     # JAX may produce ops from dialects that are not loaded by default.
    #     ctx.allow_unregistered_dialects = True
    #     module = ir.Module.parse(stablehlo_module_str)

    #     # Find 'main' function and update it.
    #     main_func_op = None
    #     for op in module.body.operations:
    #         if isinstance(op,
    #                       func_dialect.FuncOp) and op.sym_name.value == 'main':
    #             main_func_op = op
    #             break

    #     if main_func_op:
    #         breakpoint()
    #         entry_block = main_func_op.entry_block
    #         old_args = entry_block.arguments

    #     stablehlo_module_str = str(module)

    # Save the StableHLO MLIR to a file.
    mlir_path = os.path.join(script_dir, "adamw.mlir")
    with open(mlir_path, "w") as f:
        f.write(stablehlo_module_str)
    print(f"StableHLO has been written to {mlir_path}")
    # 4. Serialize all inputs to separate binary files for verification.
    local_vars = locals()
    inputs_to_save = {name: local_vars[name] for name in arg_names}
    for name, array in inputs_to_save.items():
        numpy_array = np.array(array)
        file_path = os.path.join(script_dir, f"{name}.bin")
        numpy_array.tofile(file_path)

    # 5. Run the JIT-compiled kernel to get the output for verification.
    jitted_kernel = jax.jit(adamw_kernel, donate_argnums=donate_argnums)
    updated_values = jitted_kernel(params, grads, m, v, step, lr, b1, b2, eps,
                                   weight_decay)

    # 6. Serialize all outputs to separate binary files for verification.
    # Reconstruct output names from donated args (via inspect) and other known
    # outputs. The return order from adamw_kernel is (params, m, v, step).
    donated_arg_names = [arg_names[i] for i in donate_argnums]
    output_names = donated_arg_names + ["step"]

    outputs_to_save = {
        f"{name}_new": value
        for name, value in zip(output_names, updated_values)
    }

    # For consistency, re-assign the main variables to the updated values.
    params, m, v, step = updated_values

    for name, array in outputs_to_save.items():
        numpy_array = np.array(array)
        file_path = os.path.join(script_dir, f"{name}.bin")
        numpy_array.tofile(file_path)

    print(
        f"Inputs and outputs have been written to binary files in {script_dir}"
    )

    # 7. Generate a signature file for C++ to read.
    arg_specs = {
        "params": e_params,
        "grads": e_grads,
        "m": e_m,
        "v": e_v,
        "step": e_step,
        "lr": e_lr,
        "b1": e_b1,
        "b2": e_b2,
        "eps": e_eps,
        "weight_decay": e_weight_decay,
    }

    dynamic_dims = {
        "batch_size": "?",
        "emb_dim": "?",
    }

    inputs_signature = get_signature_list(arg_names, arg_specs, dynamic_dims)
    outputs_signature = get_signature_list(output_names, arg_specs,
                                           dynamic_dims)

    signature_data = {
        "inputs": inputs_signature,
        "outputs": outputs_signature,
        "dynamic_dims": dynamic_dims
    }
    signature_path = os.path.join(script_dir, "adamw_signature.yaml")
    with open(signature_path, "w") as f:
        yaml.dump(signature_data, f, default_flow_style=False, indent=2)

    print(f"Signature file has been written to {signature_path}")


if __name__ == "__main__":
    main()
