import gzip
import hashlib
import io
import tarfile
import pytest
from scripts.remote_metrics import unpack_input


def test_remote_inputs_are_bound_to_member_and_archive_hashes(tmp_path):
    run_id='a'*16
    incoming=tmp_path/'incoming';incoming.mkdir()
    payload=b'checkpoint bytes'
    archive=incoming/(run_id+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        info=tarfile.TarInfo(run_id+'/final.pt');info.size=len(payload)
        tar.addfile(info,io.BytesIO(payload))
    arrays=incoming/'analysis_data.npz';arrays.write_bytes(b'array bytes')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    index=dict(runs=[dict(run_id=run_id,asset=archive.name,archive_sha256=sha(archive),
                          files={'final.pt':hashlib.sha256(payload).hexdigest()})],
               analysis_data=dict(asset=arrays.name,sha256=sha(arrays)))
    run,_=unpack_input(index,run_id,incoming,tmp_path/'work')
    assert (run/'final.pt').read_bytes()==payload
    arrays.write_bytes(b'changed arrays')
    with pytest.raises(ValueError,match='checksum mismatch'):
        unpack_input(index,run_id,incoming,tmp_path/'other')
