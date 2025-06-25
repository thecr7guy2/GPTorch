from ldap3 import Server, Connection, Tls
import ssl

def test_direct_bind(uid, password):
    ldap_url = "ldaps://10.1.105.10:636"
    user_dn = f"uid={uid},ou=people,dc=astnetwork,dc=local"

    tls_config = Tls(
        validate=ssl.CERT_NONE,
        version=ssl.PROTOCOL_TLSv1_2,
        ciphers="ALL"
    )

    server = Server(ldap_url, use_ssl=True, tls=tls_config)

    try:
        conn = Connection(server, user=user_dn, password=password, auto_bind=True)
        print(f"✅ Bind successful for user: {uid}")
        conn.unbind()
    except Exception as e:
        print(f"❌ Bind failed for user: {uid}")
        print(f"Error: {e}")

# 🔧 Replace with your own UID and password
test_direct_bind("ms", "redacted")
