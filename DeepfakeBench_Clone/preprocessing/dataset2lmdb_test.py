import os
import json
import cv2
import lmdb
import yaml
import argparse
import numpy as np

# ---- kleine Hilfen ----
ALLOWED = (".png", ".jpg", ".jpeg", ".npy")

def file_to_binary(file_path):
    if file_path.endswith('.npy'):
        data = np.load(file_path)
        return data.tobytes()
    else:
        with open(file_path, 'rb') as f:
            return f.read()

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _open_env(lmdb_path, map_size):
    return lmdb.open(
        lmdb_path,
        map_size=map_size,
        subdir=True,
        lock=True,
        readahead=False,
        writemap=False,
        max_readers=126,
    )

# ---- Ordner-Modus (altes Verhalten) -> KEYS: dataset_name/relpath ----
def create_lmdb_from_folder(source_folder, lmdb_path, dataset_name, map_size):
    _ensure_dir(lmdb_path)
    env = _open_env(lmdb_path, map_size)
    written, BATCH = 0, 2000
    txn = env.begin(write=True)

    def norm_key(path: str) -> bytes:
        rel = os.path.relpath(path, source_folder).replace(os.sep, "/")
        return f"{dataset_name}/{rel}".encode("utf-8")

    try:
        for root, _, files in os.walk(source_folder, followlinks=True):
            files = [f for f in files if f.lower().endswith(ALLOWED)]
            for file in files:
                p = os.path.join(root, file)
                try:
                    val = file_to_binary(p)
                except Exception as e:
                    print(f"[WARN] skip {p}: {e}")
                    continue

                key = norm_key(p)
                try:
                    txn.put(key, val)
                except lmdb.MapFullError:
                    txn.abort()
                    curr = env.info()["map_size"]
                    env.set_mapsize(int(curr * 2))
                    txn = env.begin(write=True)
                    txn.put(key, val)

                written += 1
                if written % BATCH == 0:
                    txn.commit()
                    txn = env.begin(write=True)
        txn.commit()
    finally:
        env.sync(); env.close()

    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, subdir=True)
    stat = env.stat()
    print(f"[OK] LMDB geschrieben: {lmdb_path}  | Einträge: {stat.get('entries', 0)}")
    env.close()

# ---- JSON-Modus (OSINT) -> KEYS: ABSOLUTER PFAD (kompatibel zu abstract_dataset) ----
def _frames_from_osint_json(j, ds_key="OSINT", splits=("test","val")):
    assert ds_key in j, f"Top-Level-Key '{ds_key}' nicht in JSON."
    for lab in ("OSINT_REAL", "OSINT_FAKE"):
        labnode = j[ds_key].get(lab, {})
        for split in splits:
            splitnode = labnode.get(split, {})
            for _, info in splitnode.items():
                for f in info.get("frames", []):
                    yield os.path.abspath(f)

def create_lmdb_from_osint_json(json_path, lmdb_path, map_size, ds_key="OSINT", splits=("test","val")):
    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    paths = sorted(set(_frames_from_osint_json(j, ds_key=ds_key, splits=splits)))
    assert paths, f"Keine Frames aus {json_path} für ds_key={ds_key}, splits={splits} gefunden."

    _ensure_dir(lmdb_path)
    env = _open_env(lmdb_path, map_size)
    written, BATCH = 0, 2000
    txn = env.begin(write=True)

    try:
        for i, p in enumerate(paths, 1):
            if not os.path.isfile(p):
                print(f"[WARN] existiert nicht: {p}")
                continue
            if not p.lower().endswith(ALLOWED):
                continue

            # PNG als binär (kompakt & decodierbar) speichern
            if p.lower().endswith(".png"):
                val = file_to_binary(p)
            elif p.lower().endswith((".jpg", ".jpeg")):
                img = cv2.imread(p)
                if img is None:
                    print(f"[WARN] cv2.imread None: {p}")
                    continue
                ok, buf = cv2.imencode(".png", img)
                if not ok:
                    print(f"[WARN] imencode failed: {p}")
                    continue
                val = buf.tobytes()
            elif p.lower().endswith(".npy"):
                val = file_to_binary(p)
            else:
                continue

            key = os.path.abspath(p).encode("utf-8")  # <<<< WICHTIG: ABSOLUTER PFAD ALS KEY
            try:
                txn.put(key, val)
            except lmdb.MapFullError:
                txn.abort()
                curr = env.info()["map_size"]
                env.set_mapsize(int(curr * 2))
                txn = env.begin(write=True)
                txn.put(key, val)

            written += 1
            if written % BATCH == 0:
                txn.commit()
                txn = env.begin(write=True)

        txn.commit()
    finally:
        env.sync(); env.close()

    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, subdir=True)
    stat = env.stat()
    print(f"[OK] OSINT-LMDB geschrieben: {lmdb_path}  | Einträge: {stat.get('entries', 0)}")
    env.close()

# ---- CLI ----
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_size', type=int, default=25, required=True,
                        help='LMDB-Groesse in GB (wird als map_size vorbelegt)')
    parser.add_argument('--mode', choices=['folder','osint_json'], default='folder',
                        help='folder: rekursiv aus Ordner schreiben; osint_json: aus OSINT.json mit absoluten Keys')
    parser.add_argument('--source_folder', type=str, default=None,
                        help='nur für mode=folder: Pfad zum Quell-Ordner (rgb-Dateien)')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='nur für mode=folder: dataset_name fuer Key-Präfix')
    parser.add_argument('--json_path', type=str, default=None,
                        help='nur für mode=osint_json: Pfad zur OSINT.json')
    parser.add_argument('--ds_key', type=str, default='OSINT',
                        help='Top-Level-Key in der JSON (Standard: OSINT)')
    parser.add_argument('--splits', type=str, default='test,val',
                        help='Komma-separiert, z.B. "test,val" oder "test"')
    parser.add_argument('--lmdb_path', type=str, required=True,
                        help='Zielordner der LMDB (endet auf *_lmdb)')
    parser.add_argument('--config', type=str, default='./preprocessing/config.yaml',
                        help='optional: wird nur für Defaults im folder-Modus gelesen')
    args = parser.parse_args()

    map_size = int(args.dataset_size) * 1024 * 1024 * 1024
    splits = tuple(s.strip() for s in args.splits.split(',') if s.strip())

    if args.mode == 'folder':
        # Defaults ggf. aus YAML lesen
        if (args.source_folder is None) or (args.dataset_name is None):
            try:
                with open(args.config, 'r') as f:
                    cfg = yaml.safe_load(f)['to_lmdb']
                if args.source_folder is None:
                    args.source_folder = f"{cfg['dataset_root_path']['default']}/{cfg['dataset_name']['default']}"
                if args.dataset_name is None:
                    args.dataset_name = cfg['dataset_name']['default']
            except Exception as e:
                raise RuntimeError(f"Config konnte nicht geladen werden: {e}")
        create_lmdb_from_folder(args.source_folder, args.lmdb_path, args.dataset_name, map_size)

    else:  # osint_json
        assert args.json_path is not None, "--json_path ist erforderlich für mode=osint_json"
        create_lmdb_from_osint_json(args.json_path, args.lmdb_path, map_size, ds_key=args.ds_key, splits=splits)

