"""
Shared QuasarAttention import helper.

Stubs broken fla package __init__.py files and imports only what's needed
for QuasarAttention to work. Used by benchmark, validation, and setup scripts.
"""

import sys
import os
import types


def import_quasar_attention():
    """Import QuasarAttention with robust stubbing for broken fla modules."""

    # First try direct import
    try:
        from fla.layers.quasar import QuasarAttention
        return QuasarAttention
    except (ImportError, AttributeError, AssertionError, RuntimeError):
        pass

    # Clean up failed partial imports
    for key in list(sys.modules):
        if key == "fla" or key.startswith("fla."):
            del sys.modules[key]

    # Find fla package root
    import importlib.util
    fla_spec = importlib.util.find_spec("fla")
    if fla_spec is None:
        raise ImportError(
            "Cannot find fla package. Run setup_miner.sh first or "
            "install flash-linear-attention in editable mode."
        )
    fla_root = os.path.dirname(fla_spec.origin)

    # Stub packages whose __init__.py eagerly imports everything
    # (many submodules fail due to torch version mismatches, missing triton ops, etc.)
    stub_pkgs = {
        "fla": fla_root,
        "fla.layers": os.path.join(fla_root, "layers"),
        "fla.ops": os.path.join(fla_root, "ops"),
        "fla.modules": os.path.join(fla_root, "modules"),
    }
    for pkg_name, pkg_dir in stub_pkgs.items():
        stub = types.ModuleType(pkg_name)
        stub.__path__ = [pkg_dir]
        stub.__file__ = os.path.join(pkg_dir, "__init__.py")
        stub.__package__ = pkg_name
        sys.modules[pkg_name] = stub

    # Wire parent-child attributes
    sys.modules["fla"].layers = sys.modules["fla.layers"]
    sys.modules["fla"].ops = sys.modules["fla.ops"]
    sys.modules["fla"].modules = sys.modules["fla.modules"]

    # Import the specific modules that QuasarAttention needs from fla.modules
    from fla.modules.convolution import ShortConvolution
    from fla.modules.fused_norm_gate import FusedRMSNormGated

    # Attach to stub so `from fla.modules import X` works inside quasar.py
    sys.modules["fla.modules"].ShortConvolution = ShortConvolution
    sys.modules["fla.modules"].FusedRMSNormGated = FusedRMSNormGated

    # Now import QuasarAttention
    from fla.layers.quasar import QuasarAttention
    return QuasarAttention
