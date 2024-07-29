import os
from typing import Generator
from dialtone import Dialtone
from dialtone.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    ProviderConfig,
    RouterModelConfig,
    Provider,
)

api_keys = {
    "dialtone": os.getenv("DIALTONE_API_KEY") or "",
    "openai": os.getenv("OPENAI_API_KEY") or "",
    "anthropic": os.getenv("ANTHROPIC_API_KEY") or "",
    "google": os.getenv("GOOGLE_API_KEY") or "",
    "groq": os.getenv("GROQ_API_KEY") or "",
    "cohere": os.getenv("CO_API_KEY") or "",
    "fireworks": os.getenv("FIREWORKS_API_KEY") or "",
    "deepinfra": os.getenv("DEEPINFRA_API_KEY") or "",
    "together": os.getenv("TOGETHER_API_KEY") or "",
    "replicate": os.getenv("REPLICATE_API_KEY") or "",
}


def test_chat_completion():
    global api_keys

    """
    Test Chat Completion API
    """
    dial_configs = [
        {"quality": 1, "cost": 0},
        {"quality": 0.75, "cost": 0.25},
        {"quality": 0.5, "cost": 0.5},
        {"quality": 0.25, "cost": 0.75},
        {"quality": 0, "cost": 1},
    ]

    print()
    print("Testing Chat Completion")
    print("-----------------------")
    for dials in dial_configs:
        dialtone = Dialtone(
            api_key=api_keys["dialtone"],
            provider_config=ProviderConfig(
                openai=ProviderConfig.OpenAI(api_key=api_keys["openai"]),
                anthropic=ProviderConfig.Anthropic(api_key=api_keys["anthropic"]),
                google=ProviderConfig.Google(api_key=api_keys["google"]),
                groq=ProviderConfig.Groq(api_key=api_keys["groq"]),
                cohere=ProviderConfig.Cohere(api_key=api_keys["cohere"]),
                fireworks=ProviderConfig.Fireworks(api_key=api_keys["fireworks"]),
                deepinfra=ProviderConfig.DeepInfra(api_key=api_keys["deepinfra"]),
                together=ProviderConfig.Together(api_key=api_keys["together"]),
                replicate=ProviderConfig.Replicate(api_key=api_keys["replicate"]),
            ),
            router_model_config=RouterModelConfig(
                llama_3_70b=RouterModelConfig.Llama_3_70B(
                    no_tools_providers=[Provider.Fireworks, Provider.Groq],
                )
            ),
            dials=dials,
            base_url="http://localhost:8000",
        )

        response = dialtone.chat.completions.create(
            messages=[ChatMessage(role="user", content="Hello, world!")]
        )

        assert isinstance(response, ChatCompletion)

        choice = response.choices[0]
        assert choice.message.content is not None
        assert choice.message.role == "assistant"
        assert response.model is not None
        assert response.provider is not None

        usage = response.usage
        assert usage.prompt_tokens is not None
        assert usage.completion_tokens is not None
        assert usage.total_tokens is not None

        print(
            f"Dials: {dials} | Model: {response.model} | Provider: {response.provider}"
        )


def test_chat_completion_streaming():
    global api_keys

    """
    Test Chat Completion API (streaming)
    """
    dial_configs = [
        {"quality": 0.5, "cost": 0.5},
    ]

    print()
    print("Testing Chat Completion Streaming")
    print("---------------------------------")
    for dials in dial_configs:
        dialtone = Dialtone(
            api_key=api_keys["dialtone"],
            provider_config=ProviderConfig(
                openai=ProviderConfig.OpenAI(api_key=api_keys["openai"]),
                anthropic=ProviderConfig.Anthropic(api_key=api_keys["anthropic"]),
                google=ProviderConfig.Google(api_key=api_keys["google"]),
                groq=ProviderConfig.Groq(api_key=api_keys["groq"]),
                cohere=ProviderConfig.Cohere(api_key=api_keys["cohere"]),
                fireworks=ProviderConfig.Fireworks(api_key=api_keys["fireworks"]),
                deepinfra=ProviderConfig.DeepInfra(api_key=api_keys["deepinfra"]),
                together=ProviderConfig.Together(api_key=api_keys["together"]),
                replicate=ProviderConfig.Replicate(api_key=api_keys["replicate"]),
            ),
            router_model_config=RouterModelConfig(
                llama_3_70b=RouterModelConfig.Llama_3_70B(
                    no_tools_providers=[Provider.Fireworks, Provider.Groq],
                )
            ),
            dials=dials,
            base_url="http://localhost:8000",
        )

        response = dialtone.chat.completions.create(
            messages=[
                ChatMessage(
                    role="user",
                    content="Hey, what's up? Add many newlines to your response.",
                )
            ],
            stream=True,
        )

        assert isinstance(response, Generator)

        for chat_completion_chunk in response:
            print(chat_completion_chunk)
            assert isinstance(chat_completion_chunk, ChatCompletionChunk)


def test_chat_route():
    global api_keys

    """
    Test Chat Completion API
    """
    dial_configs = [
        {"quality": 1, "cost": 0},
        {"quality": 0.75, "cost": 0.25},
        {"quality": 0.5, "cost": 0.5},
        {"quality": 0.25, "cost": 0.75},
        {"quality": 0, "cost": 1},
    ]

    print()
    print("Testing Chat Route")
    print("-----------------")
    for dials in dial_configs:
        dialtone = Dialtone(
            api_key=api_keys["dialtone"],
            provider_config=ProviderConfig(
                openai=ProviderConfig.OpenAI(api_key=api_keys["openai"]),
                anthropic=ProviderConfig.Anthropic(api_key=api_keys["anthropic"]),
                google=ProviderConfig.Google(api_key=api_keys["google"]),
                groq=ProviderConfig.Groq(api_key=api_keys["groq"]),
                cohere=ProviderConfig.Cohere(api_key=api_keys["cohere"]),
                fireworks=ProviderConfig.Fireworks(api_key=api_keys["fireworks"]),
                deepinfra=ProviderConfig.DeepInfra(api_key=api_keys["deepinfra"]),
                together=ProviderConfig.Together(api_key=api_keys["together"]),
                replicate=ProviderConfig.Replicate(api_key=api_keys["replicate"]),
            ),
            router_model_config=RouterModelConfig(
                llama_3_70b=RouterModelConfig.Llama_3_70B(
                    no_tools_providers=[Provider.Fireworks, Provider.Groq],
                )
            ),
            dials=dials,
            base_url="http://localhost:8000",
        )

        response = dialtone.chat.route(
            messages=[ChatMessage(role="user", content="Hello, world!")]
        )

        assert response.model is not None
        assert response.quality_predictions is not None
        assert response.routing_strategy is not None
        assert response.providers is not None

        print(f"Dials: {dials} | Model: {response.model}")
