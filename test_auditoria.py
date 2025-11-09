# test_auditoria.py
import pytest, os, json
from sign_and_publish_hsm import sign_log, verify_signed_log, generate_example_log

@pytest.fixture(scope="session")
def schema_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("cfg") / "noos_log_schema.json"
    minimal = {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "title": "Registro Test", "type": "object",
      "properties": {"alpha_target": {"type": "number"}, "checksum": {"type": "string"}},
      "required": ["alpha_target", "checksum"]
    }
    p.write_text(json.dumps(minimal))
    return str(p)

@pytest.fixture
def valid_log_instance():
    return generate_example_log(alpha=0.10)

def test_01_end_to_end_validation_and_signature(valid_log_instance, schema_file, monkeypatch):
    # Force schema path env to fixture
    from sign_and_publish_hsm import sign_log, verify_signed_log
    signed = sign_log(valid_log_instance, schema_path=schema_file)
    assert 'signature' in signed
    assert verify_signed_log(signed) is True

def test_02_rejection_on_payload_tampering(valid_log_instance, schema_file):
    signed = sign_log(valid_log_instance, schema_path=schema_file)
    signed['log_content']['alpha_target'] = 0.99
    assert verify_signed_log(signed) is False

def test_03_rejection_on_hash_tampering(valid_log_instance, schema_file):
    signed = sign_log(valid_log_instance, schema_path=schema_file)
    signed['sha256_payload'] = "0"*len(signed['sha256_payload'])
    assert verify_signed_log(signed) is False

def test_04_schema_validation_rejection(valid_log_instance, schema_file):
    del valid_log_instance['checksum']
    with pytest.raises(Exception):
        sign_log(valid_log_instance, schema_path=schema_file)
