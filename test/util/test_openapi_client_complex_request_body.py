# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import json
from typing import List

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from haystack_experimental.util.openapi import OpenAPIServiceClient, ClientConfiguration
from test.util.conftest import FastAPITestClient


class Customer(BaseModel):
    name: str
    email: str


class OrderItem(BaseModel):
    product: str
    quantity: int


class Order(BaseModel):
    customer: Customer
    items: List[OrderItem]


class OrderResponse(BaseModel):
    orderId: str  # noqa: N815
    status: str
    totalAmount: float  # noqa: N815


def create_order_app() -> FastAPI:
    app = FastAPI()

    @app.post("/orders")
    def create_order(order: Order):
        total_amount = sum(item.quantity * 10 for item in order.items)
        response = OrderResponse(
            orderId="ORDER-001",
            status="CREATED",
            totalAmount=total_amount,
        )
        return JSONResponse(content=response.model_dump(), status_code=201)

    return app


class TestComplexRequestBody:

    @pytest.mark.parametrize("spec_file_path", ["openapi_order_service.yml", "openapi_order_service.json"])
    def test_create_order(self, spec_file_path, test_files_path):
        path_element = "yaml" if spec_file_path.endswith(".yml") else "json"

        config = ClientConfiguration(openapi_spec=test_files_path / path_element / spec_file_path,
                                     request_sender=FastAPITestClient(create_order_app()))

        client = OpenAPIServiceClient(config)
        order_json = {
            "customer": {"name": "John Doe", "email": "john@example.com"},
            "items": [
                {"product": "Product A", "quantity": 2},
                {"product": "Product B", "quantity": 1},
            ],
        }
        payload = {
            "id": "call_NJr1NBz2Th7iUWJpRIJZoJIA",
            "function": {
                "arguments": json.dumps(order_json),
                "name": "createOrder",
            },
            "type": "function",
        }
        response = client.invoke(payload)
        assert response == {
            "orderId": "ORDER-001",
            "status": "CREATED",
            "totalAmount": 30,
        }
