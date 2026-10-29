"""Reconstruct the frozen BoT-IoT row selection and training-only transforms."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
EXPECTED = {
    "UNSW_2018_IoT_Botnet_Full5pc_1.csv": (272335105, "3940b549851ac2ce0335f3075c8308b060ecfea8973b104060e5e30867c438e3"),
    "UNSW_2018_IoT_Botnet_Full5pc_2.csv": (273082424, "1f87b8f1528a3357c9fb74d9d728d798f1996441805c668675cb82403c73d168"),
    "UNSW_2018_IoT_Botnet_Full5pc_3.csv": (282131555, "3f05d6827ece70b38d4fd90a9ae3a94a549bc8884fd990be1326f263cc29a747"),
    "UNSW_2018_IoT_Botnet_Full5pc_4.csv": (189577740, "ba7f6b36638246d0fde33a0ba31a0d243bfa700b1e71aebfae7e0346c076a8c9"),
}
CLASSES = ["DDoS", "DoS", "Reconnaissance", "Normal", "Theft"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
FEATURES = [
    "flgs", "flgs_number", "proto", "proto_number", "sport", "dport", "pkts", "bytes", "state",
    "state_number", "dur", "mean", "stddev", "sum", "min", "max", "spkts", "dpkts", "sbytes",
    "dbytes", "rate", "srate", "drate", "TnBPSrcIP", "TnBPDstIP", "TnP_PSrcIP", "TnP_PDstIP",
    "TnP_PerProto", "TnP_Per_Dport", "AR_P_Proto_P_SrcIP", "AR_P_Proto_P_DstIP",
    "N_IN_Conn_P_DstIP", "N_IN_Conn_P_SrcIP", "AR_P_Proto_P_Sport", "AR_P_Proto_P_Dport",
    "Pkts_P_State_P_Protocol_P_DestIP", "Pkts_P_State_P_Protocol_P_SrcIP",
]
CATEGORICAL = ["flgs", "proto", "state"]
NUMERIC_PORTS = ["sport", "dport"]
SEEDS = [104729, 130363, 155921, 196613, 216091, 262147, 300007, 350377, 399989, 448451]
METHODS = ["FedAvg-full", "Prevalence-FCM-DCA", "RASC-FedAvg", "RASC-FCM-DCA"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def verify_files() -> list[dict]:
    records = []
    for path in FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
        size, digest = EXPECTED[path.name]
        actual = sha256_file(path)
        if path.stat().st_size != size or actual != digest:
            raise RuntimeError(f"official-byte verification failed for {path.name}")
        records.append({"file": path.name, "bytes": size, "sha256": actual})
    return records


def evenly_spaced(values: np.ndarray, cap: int) -> np.ndarray:
    if len(values) <= cap:
        return values.copy()
    positions = np.linspace(0, len(values) - 1, cap, dtype=np.int64)
    return values[positions]


def build_split_indices(train_cap=10000) -> dict[str, np.ndarray]:
    cache_path = CACHE / "split_indices.npz"
    if cache_path.exists():
        loaded = np.load(cache_path)
        return {key: loaded[key] for key in ("train", "validation", "test")}
    indices = {c: [] for c in CLASSES}; times = {c: [] for c in CLASSES}
    offset = 0
    for path in FILES:
        for chunk in pd.read_csv(path, usecols=["stime", "category"], chunksize=250_000, low_memory=False):
            cats = chunk["category"].astype(str).str.strip().to_numpy()
            t = pd.to_numeric(chunk["stime"], errors="coerce").to_numpy(float)
            global_indices = np.arange(offset, offset + len(chunk), dtype=np.int64)
            for c in CLASSES:
                keep = cats == c
                if keep.any():
                    indices[c].append(global_indices[keep]); times[c].append(t[keep])
            offset += len(chunk)

    split = {"train": [], "validation": [], "test": []}
    caps = {"train": train_cap, "validation": 2_500, "test": 10_000}
    for c in CLASSES:
        idx = np.concatenate(indices[c]); t = np.concatenate(times[c])
        order = np.lexsort((idx, t)); ordered = idx[order]
        n70, n80 = int(np.floor(.70 * len(ordered))), int(np.floor(.80 * len(ordered)))
        segments = {"train": ordered[:n70], "validation": ordered[n70:n80], "test": ordered[n80:]}
        for name, values in segments.items():
            split[name].append(evenly_spaced(values, caps[name]))
    result = {name: np.sort(np.concatenate(parts)) for name, parts in split.items()}
    np.savez_compressed(cache_path, **result)
    return result


def load_selected_rows(split_indices: dict[str, np.ndarray]) -> pd.DataFrame:
    cache_path = CACHE / "selected_rows.pkl"
    if cache_path.exists():
        return pd.read_pickle(cache_path)
    all_selected = np.sort(np.concatenate(list(split_indices.values())))
    usecols = ["stime", "category"] + FEATURES
    frames = []; offset = 0; pointer = 0
    for path in FILES:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000, low_memory=False):
            end = offset + len(chunk)
            left = np.searchsorted(all_selected, offset, side="left", sorter=None)
            right = np.searchsorted(all_selected, end, side="left", sorter=None)
            chosen = all_selected[left:right]
            if chosen.size:
                local = chosen - offset
                selected = chunk.iloc[local].copy()
                selected["__global_index"] = chosen
                frames.append(selected)
            offset = end; pointer = right
    if pointer != len(all_selected):
        raise RuntimeError("not all selected indices were recovered")
    result = pd.concat(frames, ignore_index=True)
    result.to_pickle(cache_path)
    return result


def preprocess(rows: pd.DataFrame, split_indices: dict[str, np.ndarray]):
    index_to_split = {}
    for split, idx in split_indices.items():
        index_to_split.update((int(i), split) for i in idx)
    rows = rows.copy()
    rows["__split"] = rows["__global_index"].map(index_to_split)
    rows["category"] = rows["category"].astype(str).str.strip()
    train = rows[rows["__split"] == "train"]
    transformed = pd.DataFrame(index=rows.index)
    encoder = {}
    for col in FEATURES:
        if col in CATEGORICAL:
            values = train[col].fillna("<NA>").astype(str).str.strip()
            frequencies = (values.value_counts(dropna=False) / len(values)).to_dict()
            encoder[col] = {str(k): float(v) for k, v in frequencies.items()}
            transformed[col] = rows[col].fillna("<NA>").astype(str).str.strip().map(frequencies).fillna(0.0)
        else:
            numeric = pd.to_numeric(rows[col], errors="coerce")
            if col in NUMERIC_PORTS:
                raw = rows[col].astype(str).str.strip()
                hexadecimal = raw.str.match(r"^0[xX][0-9a-fA-F]+$")
                if hexadecimal.any():
                    numeric.loc[hexadecimal] = raw.loc[hexadecimal].map(lambda value: int(value, 16))
            transformed[col] = numeric
    transformed = transformed.replace([np.inf, -np.inf], np.nan)
    train_values = transformed.loc[train.index]
    medians = train_values.median(axis=0)
    transformed = transformed.fillna(medians)
    low = train_values.fillna(medians).quantile(.005); high = train_values.fillna(medians).quantile(.995)
    transformed = transformed.clip(lower=low, upper=high, axis=1)
    center = transformed.loc[train.index].median(axis=0)
    q25 = transformed.loc[train.index].quantile(.25); q75 = transformed.loc[train.index].quantile(.75)
    scale = (q75 - q25).where((q75 - q25).abs() > 1e-12, 1.0)
    transformed = ((transformed - center) / scale).clip(-12.0, 12.0)
    output = {}
    for split in ("train", "validation", "test"):
        loc = rows["__split"] == split
        sub = rows.loc[loc]
        order = np.lexsort((sub["__global_index"].to_numpy(), sub["stime"].to_numpy(),
                            sub["category"].map(CLASS_TO_ID).to_numpy()))
        output[split] = (
            transformed.loc[loc].to_numpy(np.float64)[order],
            sub["category"].map(CLASS_TO_ID).to_numpy(np.int64)[order],
            sub["__global_index"].to_numpy(np.int64)[order],
        )
    prep = {
        "categorical_frequency_encoder": encoder,
        "median": medians.to_dict(), "winsor_low": low.to_dict(), "winsor_high": high.to_dict(),
        "robust_center": center.to_dict(), "robust_scale": scale.to_dict(),
    }
    return output, prep


def main():
    global DATA_DIR,FILES,CACHE
    p=argparse.ArgumentParser();p.add_argument('--raw-dir',type=Path,default=ROOT/'data/raw/bot');a=p.parse_args()
    DATA_DIR=a.raw_dir;FILES=[DATA_DIR/f'UNSW_2018_IoT_Botnet_Full5pc_{i}.csv' for i in range(1,5)]
    records=verify_files()
    for name,cap in [('bot_primary',10000),('bot_expanded',50000)]:
        CACHE=ROOT/'data/cache'/name;CACHE.mkdir(parents=True,exist_ok=True)
        idx=build_split_indices(cap);rows=load_selected_rows(idx);data,prep=preprocess(rows,idx)
        np.savez_compressed(ROOT/'data'/f'{name}.npz',**{f'{s}_{k}':v for s,arrays in data.items() for k,v in zip(['x','y','ids'],arrays)})
        np.savez_compressed(ROOT/'data'/f'{name}_split.npz',**idx)
        (ROOT/'data'/f'{name}_preprocessing.json').write_text(json.dumps(prep,indent=2))
        print(name,{s:len(v[1]) for s,v in data.items()})
    (ROOT/'data/bot_raw_manifest.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()
