import httpx
import json
from typing import Any, AsyncGenerator, Generator, Type, TypeVar
from dialtone.errors import (
    APIErrorRouterDetails,
    BadRequestError,
    AuthenticationError,
    MethodNotAllowedError,
    NotFoundError,
    PreconditionFailedError,
    UnprocessableEntityError,
    RateLimitError,
    InternalServerError,
    BadGatewayError,
    ServiceUnavailableError,
    ProviderModerationError,
    ConfigurationError,
    APIError,
    ErrorCode,
    StatusCode,
)
from dialtone.types import ChatCompletionChunk
from dialtone.config import DEFAULT_REQUEST_TIMEOUT

# Generic type variable
T = TypeVar("T")


def get_status_code_from_error_code(error_code: ErrorCode) -> StatusCode:
    STATUS_CODE_FROM_CODE = {
        ErrorCode.bad_request: StatusCode.bad_request,
        ErrorCode.unauthorized: StatusCode.unauthorized,
        ErrorCode.not_found: StatusCode.not_found,
        ErrorCode.method_not_allowed: StatusCode.method_not_allowed,
        ErrorCode.precondition_failed: StatusCode.precondition_failed,
        ErrorCode.unprocessable_entity: StatusCode.unprocessable_entity,
        ErrorCode.too_many_requests: StatusCode.too_many_requests,
        ErrorCode.internal_server_error: StatusCode.internal_server_error,
        ErrorCode.bad_gateway: StatusCode.bad_gateway,
        ErrorCode.service_unavailable: StatusCode.service_unavailable,
    }

    return STATUS_CODE_FROM_CODE[error_code]


def get_error_code_from_status_code(status_code: StatusCode) -> ErrorCode:
    CODE_FROM_STATUS_CODE = {
        StatusCode.bad_request: ErrorCode.bad_request,
        StatusCode.unauthorized: ErrorCode.unauthorized,
        StatusCode.not_found: ErrorCode.not_found,
        StatusCode.method_not_allowed: ErrorCode.method_not_allowed,
        StatusCode.precondition_failed: ErrorCode.precondition_failed,
        StatusCode.unprocessable_entity: ErrorCode.unprocessable_entity,
        StatusCode.too_many_requests: ErrorCode.too_many_requests,
        StatusCode.internal_server_error: ErrorCode.internal_server_error,
        StatusCode.bad_gateway: ErrorCode.bad_gateway,
        StatusCode.service_unavailable: ErrorCode.service_unavailable,
    }

    return CODE_FROM_STATUS_CODE[status_code]


def get_error_class_from_status_code(status_code: StatusCode) -> Type[APIError]:
    ERROR_CLASS_FROM_STATUS_CODE = {
        StatusCode.bad_request: BadRequestError,
        StatusCode.unauthorized: AuthenticationError,
        StatusCode.not_found: NotFoundError,
        StatusCode.method_not_allowed: MethodNotAllowedError,
        StatusCode.precondition_failed: PreconditionFailedError,
        StatusCode.unprocessable_entity: UnprocessableEntityError,
        StatusCode.too_many_requests: RateLimitError,
        StatusCode.internal_server_error: InternalServerError,
        StatusCode.bad_gateway: BadGatewayError,
        StatusCode.service_unavailable: ServiceUnavailableError,
    }

    return ERROR_CLASS_FROM_STATUS_CODE[status_code]


def get_error_class_from_error_code(error_code: ErrorCode) -> Type[APIError]:
    ERROR_CLASS_FROM_ERROR_CODE = {
        ErrorCode.bad_request: BadRequestError,
        ErrorCode.unauthorized: AuthenticationError,
        ErrorCode.not_found: NotFoundError,
        ErrorCode.precondition_failed: PreconditionFailedError,
        ErrorCode.unprocessable_entity: UnprocessableEntityError,
        ErrorCode.too_many_requests: RateLimitError,
        ErrorCode.internal_server_error: InternalServerError,
        ErrorCode.bad_gateway: BadGatewayError,
        ErrorCode.service_unavailable: ServiceUnavailableError,
        ErrorCode.provider_moderation: ProviderModerationError,
        ErrorCode.configuration_error: ConfigurationError,
    }

    return ERROR_CLASS_FROM_ERROR_CODE[error_code]


def process_response(response: httpx.Response):
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        error_params = {
            "status_code": StatusCode(exc.response.status_code),
            "request": exc.request,
            "response": exc.response,
        }

        try:
            response_body: dict = exc.response.json()

            error_code = response_body.get("detail", {}).get("error_code")

            if error_code:
                # Happy path
                error_class = get_error_class_from_error_code(ErrorCode(error_code))

                if response_body.get("detail", {}).get("message"):
                    error_params["message"] = response_body["detail"]["message"]

                if response_body.get("detail", {}).get("router_details"):
                    error_params["router_details"] = APIErrorRouterDetails(
                        **response_body["detail"]["router_details"]
                    )

                raise error_class(**error_params) from None
            else:
                # Response body is JSON but is invalid format.
                error_class = get_error_class_from_status_code(
                    error_params["status_code"]
                )
                raise error_class(**error_params) from None
        except json.decoder.JSONDecodeError:
            # Response body is not valid JSON
            error_class = get_error_class_from_status_code(error_params["status_code"])
            if exc.response.text:
                error_params["message"] = exc.response.text

            raise error_class(**error_params) from None

    return response.json()


def dialtone_post_request(
    url: str,
    data: dict[str, Any],
    headers: dict[str, str],
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> dict:
    with httpx.Client() as client:
        response = client.post(url, json=data, headers=headers, timeout=timeout)
        return process_response(response)


async def dialtone_post_request_async(
    url: str,
    data: dict[str, Any],
    headers: dict[str, str],
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data, headers=headers, timeout=timeout)
        return process_response(response)


def dialtone_streaming_post_request(
    url: str,
    data: dict[str, Any],
    headers: dict[str, str],
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> Generator[dict, None, None]:
    with httpx.Client() as client:
        with client.stream(
            "POST", url, json=data, headers=headers, timeout=timeout
        ) as response:
            for response_chunk_text in response.iter_text():
                for line in response_chunk_text.splitlines():
                    line = line.strip()
                    if line.startswith("data:"):
                        line = line.removeprefix("data:").strip()
                    if line:
                        response_chunk_json: dict = json.loads(line)
                        yield response_chunk_json


async def dialtone_streaming_post_request_async(
    url: str,
    data: dict[str, Any],
    headers: dict[str, str],
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> AsyncGenerator[dict, None]:
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", url, json=data, headers=headers, timeout=timeout
        ) as response:
            async for response_chunk_text in response.aiter_text():
                for line in response_chunk_text.splitlines():
                    line = line.strip()
                    if line.startswith("data:"):
                        line = line.removeprefix("data:").strip()
                    if line:
                        response_chunk_json: dict = json.loads(line)
                        yield response_chunk_json


def convert_dict_to_type_stream(
    generator: Generator[dict, None, None], converter_type: Type[T]
) -> Generator[T, None, None]:
    for item in generator:
        yield converter_type(**item)


async def convert_dict_to_type_stream_async(
    generator: AsyncGenerator[dict, None], converter_type: Type[T]
) -> AsyncGenerator[T, None]:
    async for item in generator:
        yield converter_type(**item)
