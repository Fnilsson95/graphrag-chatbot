"""Tests for documentation source extraction and link formatting."""

from __future__ import annotations

import pandas as pd

from app.sources import (
    Source,
    _title_to_label,
    _url_from_text,
    extract_sources,
    format_documentation_links,
)


def test_url_from_text_extracts_kubernetes_url() -> None:
    text = (
        "URL: https://kubernetes.io/docs/concepts/workloads/pods/\n"
        "--------------------\nA Pod is the smallest deployable unit..."
    )
    assert _url_from_text(text) == (
        "https://kubernetes.io/docs/concepts/workloads/pods/"
    )


def test_title_to_label_uses_page_name() -> None:
    url = "https://kubernetes.io/docs/concepts/workloads/pods/"
    assert _title_to_label("ignored", url) == "pods"


def test_extract_sources_prefers_url_from_text_body() -> None:
    context = {
        "text_units": pd.DataFrame(
            [
                {
                    "title": "custom-title",
                    "text": (
                        "URL: https://kubernetes.io/docs/concepts/workloads/pods/\n"
                        "--------------------\nBody"
                    ),
                }
            ]
        )
    }

    sources = extract_sources(context)

    assert sources == [
        Source(
            title="custom-title",
            url="https://kubernetes.io/docs/concepts/workloads/pods/",
        )
    ]


def test_format_documentation_links_single_source() -> None:
    sources = [
        Source(
            title="kubernetes_io_docs_concepts_workloads_pods.txt",
            url="https://kubernetes.io/docs/concepts/workloads/pods/",
        )
    ]

    footer = format_documentation_links(sources)

    assert footer == (
        "\n\nSee also: [pods]"
        "(https://kubernetes.io/docs/concepts/workloads/pods/)."
    )


def test_format_documentation_links_deduplicates_urls() -> None:
    url = "https://kubernetes.io/docs/concepts/workloads/pods/"
    sources = [
        Source(title="a", url=url),
        Source(title="b", url=url),
    ]

    footer = format_documentation_links(sources)

    assert footer.count(url) == 1
