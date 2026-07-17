#!/usr/bin/env python3
"""Convert a trained Qwen affect bundle into a DJL-loadable TorchScript model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from user_affect_training import AffectHead


class ExportedAffectModel(torch.nn.Module):
    def __init__(self, encoder: SentenceTransformer, head: AffectHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        features = {"input_ids": input_ids, "attention_mask": attention_mask}
        embedding = self.encoder(features)["sentence_embedding"]
        return self.head(embedding)


def make_trace_device_agnostic(model: torch.jit.ScriptModule) -> int:
    """Replace trace-time device constants with the device of each graph's tensor input."""
    replaced = 0
    for module in model.modules():
        try:
            graph = module.forward.graph
        except AttributeError:
            continue
        tensor_input = next((value for value in graph.inputs() if str(value.type()) == "Tensor"), None)
        if tensor_input is None:
            continue
        for node in list(graph.nodes()):
            outputs = list(node.outputs())
            if (
                node.kind() != "prim::Constant"
                or len(outputs) != 1
                or outputs[0].type().kind() != "DeviceObjType"
            ):
                continue
            dynamic_device = graph.create("prim::device", [tensor_input])
            dynamic_device.output().setType(outputs[0].type())
            dynamic_device.insertBefore(node)
            outputs[0].replaceAllUsesWith(dynamic_device.output())
            node.destroy()
            replaced += 1
        torch._C._jit_pass_dce(graph)
    if replaced == 0:
        raise RuntimeError("TorchScript trace contained no device constants to generalize")
    return replaced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path("data/models/user-affect-qwen"))
    args = parser.parse_args()
    encoder = SentenceTransformer(str(args.bundle / "text_encoder"), device="cpu")
    encoder.eval()
    embedding_dimension = encoder.get_embedding_dimension()
    head = AffectHead(embedding_dimension)
    head.load_state_dict(torch.load(args.bundle / "head.pt", map_location="cpu", weights_only=True))
    head.eval()
    model = ExportedAffectModel(encoder, head).eval()
    input_ids = torch.zeros(1, 192, dtype=torch.int64)
    attention_mask = torch.ones(1, 192, dtype=torch.int64)
    traced = torch.jit.trace(model, (input_ids, attention_mask), strict=False)
    traced_temporary = args.bundle / "model.pt.trace.tmp"
    temporary = args.bundle / "model.pt.tmp"
    try:
        traced.save(str(traced_temporary))
        portable = torch.jit.load(str(traced_temporary), map_location="cpu")
        replaced_devices = make_trace_device_agnostic(portable)
        portable.save(str(temporary))
        temporary.replace(args.bundle / "model.pt")
    finally:
        traced_temporary.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    print(f"exported={args.bundle / 'model.pt'} dynamic_devices={replaced_devices}")


if __name__ == "__main__":
    main()
