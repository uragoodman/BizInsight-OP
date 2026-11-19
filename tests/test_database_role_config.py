import pytest

from app.data.database import _rls_role_statement


def test_rls_role_statement_quotes_a_valid_configured_identifier():
    assert _rls_role_statement("bizinsight_lifecycle_app") == 'SET ROLE "bizinsight_lifecycle_app"'


@pytest.mark.parametrize("role_name", [
    "",
    "app; RESET ROLE",
    'app"; DROP ROLE app; --',
    "role.with.dot",
    "a" * 64,
])
def test_rls_role_statement_rejects_invalid_or_injectable_identifiers(role_name):
    with pytest.raises(ValueError, match="DB_RLS_ROLE"):
        _rls_role_statement(role_name)
