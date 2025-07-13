import torch

def create_interleaved_tokens(zin_list, zout_list=None):
    """
    Interleave input and output tokens from prompt pairs.

    Args:
        zin_list: list of token tensors with shape (1, h, w)
        zout_list: list of token tensors with shape (1, h, w)

    Returns:
        Flat interleaved sequence: shape [T]
    """
    sequence = []

    for i, zin in enumerate(zin_list):
        if isinstance(zin, tuple):
            zin = zin[2]  # get token indices
        if zout_list is not None:
            zout = zout_list[i]
            if isinstance(zout, tuple):
                zout = zout[2]
            zin_flat = zin.view(-1)
            zout_flat = zout.view(-1)
            interleaved = torch.empty(zin_flat.numel() * 2, dtype=zin_flat.dtype, device=zin.device)
            interleaved[0::2] = zin_flat
            interleaved[1::2] = zout_flat
            sequence.append(interleaved)
        else:
            sequence.append(zin.view(-1))

    return torch.cat(sequence, dim=0)
