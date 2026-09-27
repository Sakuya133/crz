"""Adapter to unmodified, pinned author source; no redistribution license assumed."""
from __future__ import annotations
import importlib.util
import sys
import types
from pathlib import Path
from cxr_steganalysis.provenance import sha256_file

AUTHOR_COMMIT = "2c9f57be9c24753308931ee7e016c9cb2958813d"
SOURCE_HASHES = {
    "HSMNet.py": "0add74eb142f80a9446d4b4ed047ba6cdd66874ba0a1b91330fdfe92ecaca8e1",
    "modules.py": "973e1e2c51576a5156e99866ef4f74fb76d116177061eb760057d9c8cb24b57d",
    "SRM_Kernels.npy": "c53ad3a849fa64db1244152a3030c80a4d3b11fec0af81b4196d8e6dfb3db4b9",
}


def create_hsmnet(config):
    if config["data"]["normalize"] != "none":
        raise ValueError("Author HSMNet takes float32 raw 0..255; explicitly set data.normalize: none")
    root = Path(config.get("model", {}).get("author_source", "external/HSMNet"))
    if not root.is_absolute():
        root = Path(config.get("_project_root", ".")) / root
    folder = root / "src/models"
    for name, expected in SOURCE_HASHES.items():
        if not (folder / name).is_file() or sha256_file(folder / name) != expected:
            raise ValueError(f"HSMNet source missing/modified: {folder / name}; fetch pinned {AUTHOR_COMMIT}")
    package_name = "_cxr_author_hsmnet_2c9f57b"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(folder)]
        sys.modules[package_name] = package
        spec = importlib.util.spec_from_file_location(package_name + ".HSMNet", folder / "HSMNet.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    model = sys.modules[package_name + ".HSMNet"].HSMNet()
    return model
