# attestation-poc

SEV-SNP attestation between `attestation-client` (guest) and
`attestation-server` (host): evidence from `ratsd`, verified by Veraison,
every step timed.

## Flow

1. Client generates a secp256k1 keypair (`key_generation`).
2. `POST /nonce` with the public key; the server answers with 64 random bytes
   (`nonce_request`).
3. Both sides derive `SHA512(server_nonce || SHA512(pubkey))`. The client
   hands it (base64url, unpadded) to ratsd's `/ratsd/chares` with the
   `tsm-report` attester selected and gets the evidence token
   (`ratsd_evidence_collection`).
4. `POST /attest` with evidence and public key (`server_attest_request`). The
   server opens a Veraison challenge-response session on the bound nonce
   (`veraison_create_session`), submits the evidence
   (`veraison_submit_evidence`) and decodes the EAR: `submods.SEVSNP.ear.status`
   must be `affirming` and `eat_nonce` must match (`jwt_verification`).
5. The server ECIES-encrypts the config to the client's key
   (`config_encryption`); the client decrypts it (`config_decryption`).

Nonces expire after 10 minutes and are consumed by a successful `/attest`.

## Endpoints

- `POST /nonce` `{public_key}` -> `{nonce, timings}`
- `POST /attest` `{evidence, public_key}` -> `{encrypted_config, timings}`
- `GET /health` -> `{status, nonces_active}`

`public_key` is hex (uncompressed), `nonce` hex, `evidence` and
`encrypted_config` base64.

## Running

```bash
# host (Veraison at localhost:8443; --verifier-host/--verifier-port)
attestation-server --port 8000 --config-size 1M --output ./out

# guest (ratsd at localhost:8895; --ratsd-host/--ratsd-port)
attestation-client --server-url http://10.0.2.2:8000 --output ./out
```

## Artifacts

Client `--output`: `public_key.txt`, `evidence_token.bin`,
`encrypted_config.bin`, `decrypted_config.bin`, `client_times.json`
(`key_generation`, `nonce_request`, `ratsd_evidence_collection`,
`server_attest_request`, `config_decryption`, `total_client_time`).

Server `--output`: `nonce.txt`, `jwt_result.json`, `server_times.json`
(`veraison_create_session`, `veraison_submit_evidence`, `jwt_verification`,
`config_encryption`, `total_attestation`).

Timings are in seconds.
