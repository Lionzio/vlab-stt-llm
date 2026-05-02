from fastapi.testclient import TestClient

from src.main import APP_VERSION, SERVICE_NAME, app

client = TestClient(app)


class TestHealthCheck:
    """Testes para o endpoint GET /health."""

    ENDPOINT = "/health"
    EXPECTED_KEYS = {"status", "service", "version", "message"}

    def test_status_code_is_200(self) -> None:
        """Garante que o endpoint retorna HTTP 200 OK."""
        response = client.get(self.ENDPOINT)
        assert response.status_code == 200

    def test_response_contains_all_expected_keys(self) -> None:
        """Garante que o payload contém exatamente as chaves definidas no schema."""
        response = client.get(self.ENDPOINT)
        assert response.json().keys() == self.EXPECTED_KEYS

    def test_status_value_is_ok(self) -> None:
        """Garante que o campo 'status' retorna o valor 'ok'."""
        response = client.get(self.ENDPOINT)
        assert response.json()["status"] == "ok"

    def test_service_value_matches_constant(self) -> None:
        """Garante que o campo 'service' corresponde ao identificador do serviço."""
        response = client.get(self.ENDPOINT)
        assert response.json()["service"] == SERVICE_NAME

    def test_version_value_matches_constant(self) -> None:
        """Garante que o campo 'version' corresponde à versão definida na aplicação."""
        response = client.get(self.ENDPOINT)
        assert response.json()["version"] == APP_VERSION

    def test_message_value_is_correct(self) -> None:
        """Garante que o campo 'message' contém a mensagem esperada."""
        response = client.get(self.ENDPOINT)
        assert response.json()["message"] == "API operacional"

    def test_full_payload_contract(self) -> None:
        """Valida o contrato completo do payload em uma única asserção de integração."""
        response = client.get(self.ENDPOINT)
        expected_payload = {
            "status": "ok",
            "service": SERVICE_NAME,
            "version": APP_VERSION,
            "message": "API operacional",
        }
        assert response.json() == expected_payload

    def test_content_type_is_json(self) -> None:
        """Garante que o Content-Type da resposta é application/json."""
        response = client.get(self.ENDPOINT)
        assert "application/json" in response.headers["content-type"]
