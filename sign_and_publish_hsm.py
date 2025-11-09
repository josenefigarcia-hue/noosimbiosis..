# sign_and_publish_hsm.py
import json, hashlib, os, time, uuid
from jsonschema import Draft7Validator
from typing import Dict, Any
from key_vault_stub import KMS_VAULT, VAULT_KEY_ID

SCHEMA_PATH = "noos_log_schema.json"

def generate_example_log(alpha: float = 0.1):
    log_content = {
        "session_id": str(uuid.uuid4()),
        "timestamp_utc": time.time() - 3600,
        "end_timestamp_utc": time.time() - 3540,
        "alpha_target": alpha,
        "checksum_Q": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "kill_switch_triggered": False,
        "incident_flag": False,
        "duration_actual_s": 60.0,
        "telemetry": [],
        "checksum": ""
    }
    # Simulación: calc checksum interno (en práctica lo pone el hilo B)
    log_content['checksum'] = hashlib.sha256(json.dumps(log_content, sort_keys=True).encode()).hexdigest()
    return log_content

def sign_log(log_instance: Dict[str, Any], schema_path: str = SCHEMA_PATH) -> Dict[str, Any]:
    # Validación Schema
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    Draft7Validator(schema).validate(log_instance)

    # Firma vía Vault (clave privada no sale del Vault)
    pubkey_hex = KMS_VAULT.get_public_key_hex()
    payload = json.dumps(log_instance, sort_keys=True).encode()
    signature = KMS_VAULT.sign_payload(payload)
    hash_sha256 = hashlib.sha256(payload).hexdigest()

    signed_artefact = {
        "log_content": log_instance,
        "signature": signature,
        "pubkey": pubkey_hex,
        "pubkey_id": VAULT_KEY_ID,
        "sha256_payload": hash_sha256,
        "signing_timestamp": time.time()
    }

    outname = f"signed_log_{log_instance['session_id']}.json"
    with open(outname, 'w') as f:
        json.dump(signed_artefact, f, indent=2)
    return signed_artefact

def verify_signed_log(signed_artefact: Dict[str, Any]) -> bool:
    log_instance = signed_artefact["log_content"]
    signature_hex = signed_artefact["signature"]
    pubkey_hex = signed_artefact["pubkey"]
    expected_hash = signed_artefact["sha256_payload"]

    payload = json.dumps(log_instance, sort_keys=True).encode()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_hash:
        return False

    return KMS_VAULT.verify_signature(payload, signature_hex, pubkey_hex)

if __name__ == "__main__":
    if not os.path.exists(SCHEMA_PATH):
        raise SystemExit("Falta el schema JSON: noos_log_schema.json")

    log = generate_example_log(alpha=0.15)
    signed = sign_log(log)
    ok = verify_signed_log(signed)
    print("Verificación:", ok)
    print("Artefacto guardado:", f"signed_log_{log['session_id']}.json")
