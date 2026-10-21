"""Lossless inference-model codec for a frozen common input mask.

Only rows of W_dbn1 multiplying guaranteed-zero input columns are omitted.
Convolution positions and all other tensors stay unchanged. This verifies a
deployment payload; measured research runs still use full parameter vectors.
The codec is not presented as a measured network transport or DCA update
compression, because DCA's full-vector norm would need separate treatment.
"""
from __future__ import annotations
import numpy as np
from model import FixedGatedDBNCNN

def active_parameter_indices(model:FixedGatedDBNCNN,mask:np.ndarray)->np.ndarray:
    mask=np.asarray(mask,dtype=bool)
    if mask.shape!=(model.spec.feature_count,):raise ValueError('mask/schema mismatch')
    first=np.arange(model.spec.feature_count*model.spec.dbn_hidden).reshape(model.spec.feature_count,model.spec.dbn_hidden)
    return np.concatenate([first[mask].ravel(),np.arange(first.size,model.parameter_count())])

def encode(model:FixedGatedDBNCNN,mask:np.ndarray)->bytes:
    return model.parameter_vector()[active_parameter_indices(model,mask)].astype('<f8').tobytes()

def decode(payload:bytes,model:FixedGatedDBNCNN,mask:np.ndarray)->np.ndarray:
    ix=active_parameter_indices(model,mask);values=np.frombuffer(payload,dtype='<f8')
    if len(values)!=len(ix):raise ValueError('payload size mismatch')
    vector=np.zeros(model.parameter_count(),dtype=np.float64);vector[ix]=values
    return vector
