"""Steganographic embedding algorithms."""

from cxr_steganalysis.stego.lsb_matching import (
    LSBMatchingMetadata,
    LSBMatchingResult,
    embed_lsb_matching,
    embed_lsb_matching_file,
)

__all__ = [
    "LSBMatchingMetadata",
    "LSBMatchingResult",
    "embed_lsb_matching",
    "embed_lsb_matching_file",
]
