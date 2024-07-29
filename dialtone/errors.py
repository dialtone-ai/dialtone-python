import httpx
from enum import Enum
from pydantic import BaseModel
from dialtone.types import LLM, Provider


class ErrorCode(Enum):
    # Standard
    bad_request = "bad_request"
    unauthorized = "unauthorized"
    not_found = "not_found"
    method_not_allowed = "method_not_allowed"
    precondition_failed = "precondition_failed"
    unprocessable_entity = "unprocessable_entity"
    too_many_requests = "too_many_requests"
    internal_server_error = "internal_server_error"
    bad_gateway = "bad_gateway"
    service_unavailable = "service_unavailable"
    # Custom
    provider_moderation = "provider_moderation"
    configuration_error = "configuration_error"

    def __str__(self):
        return str(self.value)


class StatusCode(Enum):
    bad_request = 400
    unauthorized = 401
    not_found = 404
    method_not_allowed = 405
    precondition_failed = 412
    unprocessable_entity = 422
    too_many_requests = 429
    internal_server_error = 500
    bad_gateway = 502
    service_unavailable = 503

    def __str__(self):
        return str(self.value)


class DialtoneError(Exception):
    pass


class APIErrorRouterDetails(BaseModel):
    model: LLM | None = None
    provider: Provider | None = None
    provider_response: dict | None = None


class APIError(DialtoneError):
    request: httpx.Request
    message: str
    router_details: APIErrorRouterDetails

    def __str__(self):
        if self.router_details.provider_response:
            return (
                f"{self.message} (model {self.router_details.model} from "
                f"{self.router_details.provider}) - Provider Response: {self.router_details.provider_response}"
            )
        return self.message


class APIStatusError(APIError):
    response: httpx.Response
    status_code: StatusCode

    def __init__(
        self,
        request: httpx.Request,
        response: httpx.Response,
        status_code: StatusCode | None = None,
        message: str | None = None,
        router_details: APIErrorRouterDetails = APIErrorRouterDetails(),
    ):
        self.request = request
        self.response = response
        self.router_details = router_details

        if status_code:
            self.status_code = status_code
        if message:
            self.message = message


class BadRequestError(APIStatusError):
    status_code: StatusCode = StatusCode.bad_request
    message: str = "Bad Request"


class AuthenticationError(APIStatusError):
    status_code: StatusCode = StatusCode.unauthorized
    message: str = "Unauthorized"


class NotFoundError(APIStatusError):
    status_code: StatusCode = StatusCode.not_found
    message: str = "Not Found"


class MethodNotAllowedError(APIStatusError):
    status_code: StatusCode = StatusCode.method_not_allowed
    message: str = "Method Not Allowed"


class PreconditionFailedError(APIStatusError):
    status_code: StatusCode = StatusCode.precondition_failed
    message: str = "Precondition Failed"


class UnprocessableEntityError(APIStatusError):
    status_code: StatusCode = StatusCode.unprocessable_entity
    message: str = "Unprocessable Entity"


class RateLimitError(APIStatusError):
    status_code: StatusCode = StatusCode.too_many_requests
    message: str = "Too Many Requests"


class InternalServerError(APIStatusError):
    status_code: StatusCode = StatusCode.internal_server_error
    message: str = "Internal Server Error"


class BadGatewayError(APIStatusError):
    status_code: StatusCode = StatusCode.bad_gateway
    message: str = "Bad Gateway"


class ServiceUnavailableError(APIStatusError):
    status_code: StatusCode = StatusCode.service_unavailable
    message: str = "Service Unavailable"


class ProviderModerationError(APIStatusError):
    status_code: StatusCode = StatusCode.bad_request
    message: str = "Provider Moderation Error"


class ConfigurationError(APIStatusError):
    status_code: StatusCode = StatusCode.bad_request
    message: str = "Configuration Error"
