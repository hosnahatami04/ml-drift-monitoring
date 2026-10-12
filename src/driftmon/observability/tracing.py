"""OpenTelemetry tracing setup.

Spans go to the console by default. Setting the standard
OTEL_EXPORTER_OTLP_ENDPOINT variable switches to OTLP over HTTP; export runs
in a background batch processor, so an unreachable collector costs a logged
warning, never a failed request.
"""

import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from driftmon.config import TelemetryConfig

logger = logging.getLogger(__name__)

OTLP_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"

TRACER_NAME = "driftmon"


def build_tracer_provider(config: TelemetryConfig, service_name: str) -> TracerProvider:
    provider = TracerProvider(
        resource=Resource.create({SERVICE_NAME: service_name}),
        sampler=ParentBased(TraceIdRatioBased(config.trace_sample_ratio)),
    )
    processor = _span_processor(config)
    if processor is not None:
        provider.add_span_processor(processor)
    return provider


def _span_processor(config: TelemetryConfig) -> SpanProcessor | None:
    if os.environ.get(OTLP_ENDPOINT_ENV):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        logger.info("exporting traces over OTLP to %s", os.environ[OTLP_ENDPOINT_ENV])
        return BatchSpanProcessor(OTLPSpanExporter())
    if config.trace_exporter == "console":
        return BatchSpanProcessor(ConsoleSpanExporter())
    return None


def setup_tracing(config: TelemetryConfig, service_name: str) -> TracerProvider:
    """Install the process-wide provider. Called once by long-running
    commands; library code only asks for a tracer."""
    provider = build_tracer_provider(config, service_name)
    trace.set_tracer_provider(provider)
    return provider


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)
