# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import json

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from haystack_experimental.components.tools.openapi._openapi import OpenAPIServiceClient, ClientConfiguration
from test.components.tools.openapi.conftest import FastAPITestClient, create_openapi_spec


class Identification(BaseModel):
    type: str
    number: str


class Payer(BaseModel):
    name: str
    email: str
    identification: Identification


class PaymentRequest(BaseModel):
    transaction_amount: float
    description: str
    payment_method_id: str
    payer: Payer


class PaymentResponse(BaseModel):
    transaction_id: str
    status: str
    message: str


def create_payment_app() -> FastAPI:
    app = FastAPI()

    @app.post("/new_payment")
    def process_payment(payment: PaymentRequest):
        # sanity
        assert payment.transaction_amount == 100.0
        response = PaymentResponse(
            transaction_id="TRANS-12345", status="SUCCESS", message="Payment processed successfully."
        )
        return JSONResponse(content=response.model_dump(), status_code=200)

    return app


# Write the unit test
class TestPaymentProcess:

    def test_process_payment(self, test_files_path):
        config = ClientConfiguration(openapi_spec=create_openapi_spec(test_files_path / "json" / "complex_types_openapi_service.json"),
                                     request_sender=FastAPITestClient(create_payment_app()))
        client = OpenAPIServiceClient(config)

        payment_json = {
            "transaction_amount": 100.0,
            "description": "Test Payment",
            "payment_method_id": "CARD-123",
            "payer": {
                "name": "Alice Smith",
                "email": "alice@example.com",
                "identification": {"type": "CPF", "number": "123.456.789-00"},
            },
        }
        payload = {
            "id": "call_uniqueID123",
            "function": {
                "arguments": json.dumps(payment_json),
                "name": "processPayment",
            },
            "type": "function",
        }
        response = client.invoke(payload)
        assert response == {
            "transaction_id": "TRANS-12345",
            "status": "SUCCESS",
            "message": "Payment processed successfully.",
        }
