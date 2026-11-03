# -*- coding: utf-8 -*-
"""探测 Oxford / NASA .mat 文件内部结构，仅做结构打印，不做处理。"""
import sys
import scipy.io as sio

H5 = None
try:
    import h5py
except Exception:
    h5py = None


def probe(path):
    print("=" * 70)
    print("FILE:", path)
    # 先尝试 scipy（v5 格式）
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        print("[scipy v5] keys:", [k for k in mat.keys() if not k.startswith("__")])
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            print("  -", k, type(v).__name__, getattr(v, "shape", ""), getattr(v, "dtype", ""))
            _walk(v, depth=1, maxdepth=2)
        return
    except NotImplementedError:
        print("[scipy] 不支持，疑似 v7.3(HDF5)，改用 h5py")
    except Exception as e:
        print("[scipy] 失败:", repr(e))
    if h5py is not None:
        with h5py.File(path, "r") as f:
            print("[h5py] top keys:", list(f.keys()))

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print("  D", name, obj.shape, obj.dtype)
                else:
                    print("  G", name)

            f.visititems(visit)


def _walk(v, depth, maxdepth):
    if depth > maxdepth:
        return
    # mat_struct
    if hasattr(v, "_fieldnames"):
        for fn in v._fieldnames:
            try:
                fv = getattr(v, fn)
                print(" " * (depth * 2), ".", fn, type(fv).__name__,
                      getattr(fv, "shape", ""), getattr(fv, "dtype", ""))
                _walk(fv, depth + 1, maxdepth)
            except Exception as e:
                print(" " * (depth * 2), ".", fn, "ERR", e)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            probe(p)
        except Exception as e:
            print("PROBE ERROR", p, repr(e))
