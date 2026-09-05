from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import onnxruntime as ort
import pytest

from lode.adapters import OnnxTextEmbeddingAdapter
from lode.domain import EmbeddingError


def test_constructor_stores_configuration(
    valid_model_dir: Path,
) -> None:
    with (
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_tokenizer",
            return_value=Mock(),
        ),
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_session",
        ) as create_session,
    ):
        session = Mock()
        session.get_outputs.return_value = [
            SimpleNamespace(name="last_hidden_state")
        ]
        create_session.return_value = session

        adapter = OnnxTextEmbeddingAdapter(
            valid_model_dir,
            max_sequence_length=512,
        )

    assert adapter._model_dir == valid_model_dir.resolve()
    assert adapter._max_sequence_length == 512


def test_constructor_creates_session(
    valid_model_dir: Path,
) -> None:
    with (
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_tokenizer",
            return_value=Mock(),
        ),
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_session",
        ) as create_session,
    ):
        session = Mock()
        session.get_outputs.return_value = [
            SimpleNamespace(name="last_hidden_state")
        ]
        create_session.return_value = session

        adapter = OnnxTextEmbeddingAdapter(valid_model_dir)

    create_session.assert_called_once_with(
        intra_op_threads=1,
        inter_op_threads=1,
    )

    assert adapter._session is session


def test_constructor_reads_output_name(
    valid_model_dir: Path,
) -> None:
    with (
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_tokenizer",
            return_value=Mock(),
        ),
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_session",
        ) as create_session,
    ):
        session = Mock()
        session.get_outputs.return_value = [
            SimpleNamespace(name="sentence_embedding")
        ]
        create_session.return_value = session

        adapter = OnnxTextEmbeddingAdapter(valid_model_dir)

    assert adapter._output_name == "sentence_embedding"


def test_constructor_raises_when_model_directory_does_not_exist(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "missing"

    with pytest.raises(
        EmbeddingError,
        match="Model directory does not exist",
    ):
        OnnxTextEmbeddingAdapter(model_dir)


def test_constructor_raises_when_model_path_is_not_directory(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "model"

    file_path.write_text("dummy")

    with pytest.raises(
        EmbeddingError,
        match="Model path is not a directory",
    ):
        OnnxTextEmbeddingAdapter(file_path)


@pytest.mark.parametrize(
    "missing_file",
    [
        "model.onnx",
        "tokenizer.json",
        "config.json",
        "tokenizer_config.json",
    ],
)
def test_constructor_raises_when_required_file_is_missing(
    valid_model_dir: Path,
    missing_file: str,
) -> None:
    if missing_file == "model.onnx":
        path = valid_model_dir / "onnx" / missing_file
    else:
        path = valid_model_dir / "onnx" / missing_file

    path.unlink()

    with pytest.raises(
        EmbeddingError,
        match="Required model file not found",
    ):
        OnnxTextEmbeddingAdapter(valid_model_dir)


@pytest.mark.parametrize(
    "filename",
    [
        "config.json",
        "tokenizer_config.json",
    ],
)
def test_constructor_raises_when_json_is_invalid(
    valid_model_dir: Path,
    filename: str,
) -> None:
    path = valid_model_dir / "onnx" / filename

    path.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(
        EmbeddingError,
        match="Invalid JSON file",
    ):
        OnnxTextEmbeddingAdapter(valid_model_dir)


# Model Directory

def test_build_session_options_uses_given_thread_counts() -> None:
    options = OnnxTextEmbeddingAdapter._build_session_options(
        intra_op_threads=1,
        inter_op_threads=2,
    )

    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 2


def test_build_session_options_enables_graph_optimization() -> None:
    options = OnnxTextEmbeddingAdapter._build_session_options(
        intra_op_threads=1,
        inter_op_threads=1,
    )

    assert (
        options.graph_optimization_level
        == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )


def test_build_session_options_enables_memory_pattern() -> None:
    options = OnnxTextEmbeddingAdapter._build_session_options(
        intra_op_threads=1,
        inter_op_threads=1,
    )

    assert options.enable_mem_pattern is True


def test_build_session_options_enables_cpu_memory_arena() -> None:
    options = OnnxTextEmbeddingAdapter._build_session_options(
        intra_op_threads=1,
        inter_op_threads=1,
    )

    assert options.enable_cpu_mem_arena is True


def test_create_session_builds_onnx_runtime_session(
    valid_model_dir: Path,
) -> None:
    adapter = object.__new__(OnnxTextEmbeddingAdapter)

    adapter._model_dir = valid_model_dir.resolve()

    with patch.object(
        OnnxTextEmbeddingAdapter,
        "_build_session_options",
    ) as build_options, patch(
        "lode.adapters.embedders.onnx.ort.InferenceSession",
    ) as inference_session:
        options = Mock()
        build_options.return_value = options

        adapter._create_session(
            intra_op_threads=2,
            inter_op_threads=4,
        )

    build_options.assert_called_once_with(
        intra_op_threads=2,
        inter_op_threads=4,
    )

    inference_session.assert_called_once_with(
        str(adapter._model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )



import numpy as np


def test_tokenize_pads_shorter_sequences(
    valid_model_dir: Path,
) -> None:
    tokenizer = Mock()

    tokenizer.encode_batch.return_value = [
        SimpleNamespace(
            ids=[1, 2, 3],
            type_ids=[],
        ),
        SimpleNamespace(
            ids=[4, 5],
            type_ids=[],
        ),
    ]

    session = Mock()
    session.get_outputs.return_value = [
        SimpleNamespace(name="last_hidden_state")
    ]

    with (
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_tokenizer",
            return_value=tokenizer,
        ),
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_session",
            return_value=session,
        ),
    ):
        adapter = OnnxTextEmbeddingAdapter(valid_model_dir)

    inputs = adapter._tokenize(
        (
            "first",
            "second",
        )
    )

    np.testing.assert_array_equal(
        inputs["input_ids"],
        np.asarray(
            [
                [1, 2, 3],
                [4, 5, 0],
            ],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        inputs["attention_mask"],
        np.asarray(
            [
                [1, 1, 1],
                [1, 1, 0],
            ],
            dtype=np.int64,
        ),
    )

    assert "token_type_ids" not in inputs





def test_tokenize_truncates_long_sequences(
    valid_model_dir: Path,
) -> None:
    tokenizer = Mock()

    tokenizer.encode_batch.return_value = [
        SimpleNamespace(
            ids=[1, 2, 3, 4, 5, 6],
            type_ids=[],
        ),
    ]

    session = Mock()
    session.get_outputs.return_value = [
        SimpleNamespace(name="last_hidden_state")
    ]

    with (
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_tokenizer",
            return_value=tokenizer,
        ),
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_session",
            return_value=session,
        ),
    ):
        adapter = OnnxTextEmbeddingAdapter(
            valid_model_dir,
            max_sequence_length=4,
        )

    inputs = adapter._tokenize(("example",))

    np.testing.assert_array_equal(
        inputs["input_ids"],
        np.asarray(
            [
                [1, 2, 3, 4],
            ],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        inputs["attention_mask"],
        np.asarray(
            [
                [1, 1, 1, 1],
            ],
            dtype=np.int64,
        ),
    )

    assert "token_type_ids" not in inputs


def test_tokenize_includes_token_type_ids(
    valid_model_dir: Path,
) -> None:
    tokenizer = Mock()

    tokenizer.encode_batch.return_value = [
        SimpleNamespace(
            ids=[10, 20, 30],
            type_ids=[0, 0, 1],
        ),
        SimpleNamespace(
            ids=[40, 50],
            type_ids=[0, 1],
        ),
    ]

    session = Mock()
    session.get_outputs.return_value = [
        SimpleNamespace(name="last_hidden_state")
    ]

    with (
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_tokenizer",
            return_value=tokenizer,
        ),
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_session",
            return_value=session,
        ),
    ):
        adapter = OnnxTextEmbeddingAdapter(valid_model_dir)

    inputs = adapter._tokenize(
        (
            "first",
            "second",
        )
    )

    np.testing.assert_array_equal(
        inputs["token_type_ids"],
        np.asarray(
            [
                [0, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.int64,
        ),
    )


def test_run_inference_returns_float32_output(
    valid_model_dir: Path,
) -> None:
    session = Mock()

    output = np.random.rand(2, 3, 384).astype(np.float32)

    session.run.return_value = [output]
    session.get_outputs.return_value = [
        SimpleNamespace(name="last_hidden_state")
    ]

    with (
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_tokenizer",
            return_value=Mock(),
        ),
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_session",
            return_value=session,
        ),
    ):
        adapter = OnnxTextEmbeddingAdapter(valid_model_dir)

    inputs = {
        "input_ids": np.zeros((2, 3), dtype=np.int64),
        "attention_mask": np.ones((2, 3), dtype=np.int64),
    }

    result = adapter._run_inference(inputs)

    session.run.assert_called_once_with(
        ["last_hidden_state"],
        inputs,
    )

    np.testing.assert_array_equal(result, output)

    assert result.dtype == np.float32


def test_run_inference_wraps_runtime_errors(
    valid_model_dir: Path,
) -> None:
    session = Mock()

    session.run.side_effect = RuntimeError("boom")

    session.get_outputs.return_value = [
        SimpleNamespace(name="last_hidden_state")
    ]

    with (
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_tokenizer",
            return_value=Mock(),
        ),
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_session",
            return_value=session,
        ),
    ):
        adapter = OnnxTextEmbeddingAdapter(valid_model_dir)

    inputs = {
        "input_ids": np.zeros((1, 2), dtype=np.int64),
        "attention_mask": np.ones((1, 2), dtype=np.int64),
    }

    with pytest.raises(
        EmbeddingError,
        match="ONNX inference failed",
    ):
        adapter._run_inference(inputs)


def test_mean_pool_with_full_attention_mask() -> None:
    hidden_states = np.asarray(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ],
        dtype=np.float32,
    )

    attention_mask = np.asarray(
        [
            [1, 1],
        ],
        dtype=np.int64,
    )

    pooled = OnnxTextEmbeddingAdapter._mean_pool(
        hidden_states,
        attention_mask,
    )

    expected = np.asarray(
        [
            [2.0, 3.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        pooled,
        expected,
    )


def test_mean_pool_ignores_padding_tokens() -> None:
    hidden_states = np.asarray(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [100.0, 100.0],
            ]
        ],
        dtype=np.float32,
    )

    attention_mask = np.asarray(
        [
            [1, 1, 0],
        ],
        dtype=np.int64,
    )

    pooled = OnnxTextEmbeddingAdapter._mean_pool(
        hidden_states,
        attention_mask,
    )

    expected = np.asarray(
        [
            [2.0, 3.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        pooled,
        expected,
    )


def test_mean_pool_handles_batch_inputs() -> None:
    hidden_states = np.asarray(
        [
            [
                [1.0],
                [3.0],
            ],
            [
                [10.0],
                [20.0],
            ],
        ],
        dtype=np.float32,
    )

    attention_mask = np.asarray(
        [
            [1, 1],
            [1, 1],
        ],
        dtype=np.int64,
    )

    pooled = OnnxTextEmbeddingAdapter._mean_pool(
        hidden_states,
        attention_mask,
    )

    expected = np.asarray(
        [
            [2.0],
            [15.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        pooled,
        expected,
    )


def test_l2_normalize_returns_unit_vectors() -> None:
    embeddings = np.asarray(
        [
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    normalized = OnnxTextEmbeddingAdapter._l2_normalize(
        embeddings,
    )

    expected = np.asarray(
        [
            [0.6, 0.8],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        normalized,
        expected,
        rtol=1e-6,
    )


def test_l2_normalize_outputs_unit_norm() -> None:
    embeddings = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )

    normalized = OnnxTextEmbeddingAdapter._l2_normalize(
        embeddings,
    )

    norms = np.linalg.norm(
        normalized,
        axis=1,
    )

    np.testing.assert_allclose(
        norms,
        np.ones_like(norms),
        rtol=1e-6,
    )


def test_l2_normalize_handles_zero_vectors() -> None:
    embeddings = np.asarray(
        [
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    normalized = OnnxTextEmbeddingAdapter._l2_normalize(
        embeddings,
    )

    np.testing.assert_array_equal(
        normalized,
        np.zeros_like(embeddings),
    )

    assert not np.isnan(normalized).any()


def test_compute_embeddings_runs_pipeline_in_order(
    valid_model_dir: Path,
) -> None:
    session = Mock()
    session.get_outputs.return_value = [
        SimpleNamespace(name="last_hidden_state")
    ]

    with (
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_tokenizer",
            return_value=Mock(),
        ),
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_session",
            return_value=session,
        ),
    ):
        adapter = OnnxTextEmbeddingAdapter(valid_model_dir)

    tokenized = {
        "attention_mask": np.asarray([[1, 1]], dtype=np.int64),
    }

    hidden_states = np.asarray(
        [[[1.0, 2.0]]],
        dtype=np.float32,
    )

    pooled = np.asarray(
        [[1.5, 2.5]],
        dtype=np.float32,
    )

    normalized = np.asarray(
        [[0.5, 0.8]],
        dtype=np.float32,
    )

    with (
        patch.object(
            adapter,
            "_tokenize",
            return_value=tokenized,
        ) as tokenize,
        patch.object(
            adapter,
            "_run_inference",
            return_value=hidden_states,
        ) as inference,
        patch.object(
            adapter,
            "_mean_pool",
            return_value=pooled,
        ) as mean_pool,
        patch.object(
            adapter,
            "_l2_normalize",
            return_value=normalized,
        ) as normalize,
    ):
        result = adapter._compute_embeddings(
            ("hello",),
        )

    tokenize.assert_called_once_with(("hello",))

    inference.assert_called_once_with(tokenized)

    mean_pool.assert_called_once_with(
        hidden_states,
        tokenized["attention_mask"],
    )

    normalize.assert_called_once_with(pooled)

    assert result is normalized


@pytest.mark.asyncio
async def test_embed_returns_empty_tuple_for_empty_input(
    valid_model_dir: Path,
) -> None:
    session = Mock()
    session.get_outputs.return_value = [
        SimpleNamespace(name="last_hidden_state")
    ]

    with (
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_tokenizer",
            return_value=Mock(),
        ),
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_session",
            return_value=session,
        ),
    ):
        adapter = OnnxTextEmbeddingAdapter(valid_model_dir)

    with patch.object(
        adapter,
        "_compute_embeddings",
    ) as compute:
        result = await adapter.embed(
            (),
            mode="document",
        )

    assert result == tuple()

    compute.assert_not_called()


@pytest.mark.asyncio
async def test_embed_applies_mode_prefix_before_embedding(
    valid_model_dir: Path,
) -> None:
    session = Mock()
    session.get_outputs.return_value = [
        SimpleNamespace(name="last_hidden_state")
    ]

    with (
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_tokenizer",
            return_value=Mock(),
        ),
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_session",
            return_value=session,
        ),
    ):
        adapter = OnnxTextEmbeddingAdapter(valid_model_dir)

    fake_embeddings = np.asarray(
        [[1.0, 2.0]],
        dtype=np.float32,
    )

    with patch.object(
        adapter,
        "_compute_embeddings",
        return_value=fake_embeddings,
    ) as compute:
        await adapter.embed(
            ("hello",),
            mode="query",
        )

    compute.assert_called_once_with(
        ("query: hello",),
    )


@pytest.mark.asyncio
async def test_embed_returns_domain_embeddings(
    valid_model_dir: Path,
) -> None:
    session = Mock()
    session.get_outputs.return_value = [
        SimpleNamespace(name="last_hidden_state")
    ]

    with (
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_tokenizer",
            return_value=Mock(),
        ),
        patch.object(
            OnnxTextEmbeddingAdapter,
            "_create_session",
            return_value=session,
        ),
    ):
        adapter = OnnxTextEmbeddingAdapter(valid_model_dir)

    fake_embeddings = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    with patch.object(
        adapter,
        "_compute_embeddings",
        return_value=fake_embeddings,
    ):
        result = await adapter.embed(
            (
                "a",
                "b",
            ),
            mode="document",
        )

    assert result == (
        (1.0, 2.0),
        (3.0, 4.0),
    )
