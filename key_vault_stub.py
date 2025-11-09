# key_vault_stub.py
import os
from nacl.signing import SigningKey, VerifyKey

VAULT_KEY_ID = "NOO_SYMBIOSIS_PHASE2_KMS_ID_V1"

class KeyVaultStub:
    """
    Stub minimal de HSM/Vault para firmas ed25519.
    En producción sustituir por AWS KMS / HashiCorp Vault / HSM.
    """
    def __init__(self):
        self._signing_key = self._load_or_generate_key()
        self._verify_key = self._signing_key.verify_key

    def _load_or_generate_key(self):
        seed_env = os.environ.get("NOOSIMBIOSIS_KEY_SEED")
        if seed_env is not None:
            seed = seed_env.encode() if isinstance(seed_env, str) else seed_env
            seed = seed.ljust(32, b'\0')[:32]
            return SigningKey(seed)
        # fallback: determinista solo para demo; NO usar en producción
        return SigningKey(b'default_seed_for_demo__________'[:32])

    def get_public_key_hex(self) -> str:
        return self._verify_key.encode().hex()

    def sign_payload(self, payload: bytes) -> str:
        """Firma el payload y devuelve la firma hex. Clave privada no exportable."""
        return self._signing_key.sign(payload).signature.hex()

    def verify_signature(self, payload: bytes, signature_hex: str, pubkey_hex: str) -> bool:
        try:
            vk = VerifyKey(bytes.fromhex(pubkey_hex))
            vk.verify(payload, bytes.fromhex(signature_hex))
            return True
        except Exception:
            return False

# Instancia global
KMS_VAULT = KeyVaultStub()
