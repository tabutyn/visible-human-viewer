import gzip
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pack_delivery import pack, pack_one


def test_lossless_rgb_and_ct_without_source_mutations(tmp_path):
    source=tmp_path/"processed"
    output=tmp_path/"delivery"
    (source/"male/rgb").mkdir(parents=True)
    (source/"male/density").mkdir()
    image=Image.new("RGB",(64,48),(210,73,42))
    image.save(source/"male/rgb/0.png",compress_level=0)
    raw=bytes(range(256))*512
    (source/"male/density/0.u16be").write_bytes(raw)
    before={p:p.read_bytes() for p in source.rglob("*") if p.is_file()}
    for relative,encoding in [("male/rgb/0.png","webp-lossless"),("male/density/0.u16be","gzip")]:
        result=pack_one(source,output,relative,encoding)
        assert result["bytes"] < result["source_bytes"]
        assert result["source_sha256"]==hashlib.sha256(before[source/relative]).hexdigest()
        packed=output/result["file"]
        if encoding=="gzip":assert gzip.decompress(packed.read_bytes())==raw
        else:
            with Image.open(packed) as decoded:assert decoded.tobytes()==image.tobytes()
    assert all(p.read_bytes()==value for p,value in before.items())


def test_manifest_resume_and_source_change(tmp_path):
    root=tmp_path/"processed"
    output=tmp_path/"delivery"
    for folder in ["volume-v1","rgb-volume-v1"]:
        directory=root/"male"/folder
        directory.mkdir(parents=True)
        (directory/"brick").write_bytes(b"abc"*4096)
        (directory/"manifest.json").write_text(json.dumps({"levels":[{"level":2,"bricks":[{"file":"brick"}]}]}))
    first=pack(root,output)
    index=json.loads((output/"index.json").read_text())
    assert first["packed_files"]==2
    assert pack(root,output)["delivery_bytes"]==first["delivery_bytes"]
    (root/"male/volume-v1/brick").write_bytes(b"xyz"*4096)
    pack(root,output)
    updated=json.loads((output/"index.json").read_text())
    assert updated["files"]["male/volume-v1/brick"]["source_sha256"]!=index["files"]["male/volume-v1/brick"]["source_sha256"]
    with pytest.raises(ValueError):pack(root,root/"bad")
    with pytest.raises(ValueError):pack_one(root,output,"../../outside","gzip")
