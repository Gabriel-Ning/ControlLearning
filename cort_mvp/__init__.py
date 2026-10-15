"""CORT-MVP: controlled study of vision-to-action transport as a representation-learning mechanism."""
import re as _re

import huggingface_hub as _hf

# huggingface_hub>=2 removed `repo_type_and_id_from_hf_id`, which timm imports at module load (only used for pushing
# models to the hub). Provide a minimal implementation BEFORE any timm import so `timm.create_model(pretrained=True)`
# works. Upstream VITA code (`flare`) imports timm, so this must live at package import time.
if not hasattr(_hf, "repo_type_and_id_from_hf_id"):
    def _repo_type_and_id_from_hf_id(hf_id: str, hub_url: str | None = None):
        hf_id = _re.sub(r"^https?://[^/]+/", "", hf_id)
        parts = hf_id.split("/")
        repo_type = None
        if parts[0] in ("models", "datasets", "spaces"):
            repo_type = parts.pop(0).rstrip("s")
        namespace = parts[0] if len(parts) > 1 else None
        return repo_type, namespace, parts[-1]
    _hf.repo_type_and_id_from_hf_id = _repo_type_and_id_from_hf_id
